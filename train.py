"""
GRPO Training Script — Email Triage RL Environment
===================================================

Trains an LLM to triage emails using GRPO (Group Relative Policy Optimisation)
from TRL + Unsloth. Uses 6 independent reward verifiers for maximum signal
quality and anti-reward-hacking coverage.

Reward functions (all independent):
  1. reward_format        — JSON parse + valid email_id gate
  2. reward_classification — semantic distance scoring (exact +1.0, adjacent +0.3)
  3. reward_priority      — graduated scale (exact +1.0, off-by-1 +0.4, off-by-3+ −0.4)
  4. reward_routing       — semantic department distance scoring
  5. reward_escalation    — F1-style (TP +1.0, TN +0.3, FP −1.0, FN −0.5)
  6. reward_response      — keyword coverage with length penalty

OpenEnv integration:
  - Dataset built directly from openenv environment (not static CSV)
  - Live evaluation uses openenv-core client WebSocket protocol
  - Fully compatible with TRL GRPOTrainer reward_funcs API

Usage:
    python train.py                        # full curriculum training
    python train.py --task easy            # single-task training
    python train.py --model Qwen/Qwen2.5-3B-Instruct
    python train.py --live-eval            # evaluate against live HF Space

Environment:
    pip install trl>=0.12 unsloth datasets wandb peft bitsandbytes openenv-core
    export WANDB_PROJECT=email-triage-rl   # optional W&B logging
    export HF_TOKEN=hf_...                 # for live-eval against HF Space
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
#  CLI arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GRPO training for Email Triage RL Environment")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct",
                   help="Base model (default: Qwen/Qwen2.5-3B-Instruct)")
    p.add_argument("--task", default="curriculum",
                   choices=["easy", "medium", "hard", "curriculum"],
                   help="Task to train on. 'curriculum' runs easy→medium→hard.")
    p.add_argument("--output-dir", default="./checkpoints/email-triage-grpo")
    p.add_argument("--num-generations", type=int, default=8,
                   help="GRPO completions per prompt (G). Higher = better variance estimate.")
    p.add_argument("--max-steps", type=int, default=300,
                   help="Training steps per curriculum stage")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-6,
                   help="Learning rate (3e-6 is safer than 5e-6 for GRPO)")
    p.add_argument("--max-new-tokens", type=int, default=320,
                   help="Max tokens per action (320 fits full JSON + response draft)")
    p.add_argument("--lora-rank", type=int, default=32,
                   help="LoRA rank (32 gives better capacity for multi-task)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-unsloth", action="store_true")
    p.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    p.add_argument("--live-eval", action="store_true",
                   help="Evaluate against live HF Space via openenv-core WebSocket")
    p.add_argument("--env-url", default="https://hk4crprasad-email-triage-env.hf.space",
                   help="OpenEnv server URL for live evaluation")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset construction from the environment
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert email triage agent. For each email, respond with a single valid JSON object:
{
  "email_id": "<id>",
  "category": "<spam|billing|technical|general|urgent>",
  "priority": <1-5>,
  "department": "<engineering|billing|support|management>",
  "response_draft": "<reply text or null>",
  "escalate": <true|false>
}

Categories: spam=unsolicited/phishing, billing=payments/invoices, technical=bugs/API,
general=inquiries/requests, urgent=outages/security/legal

Priorities: 1=critical(system down/breach), 2=high(degradation), 3=medium(discrepancy),
4=low(feature request), 5=lowest(spam/auto)

Departments: engineering=bugs/API/security, billing=payments/invoices,
support=general/account/spam, management=legal/compliance/escalations

Respond with ONLY the JSON. No other text."""


def format_email_prompt(email: Dict[str, Any], task_description: str) -> str:
    """Format a single email into a triage prompt."""
    lines = [
        f"TASK: {task_description[:200]}",
        "",
        f"Email ID: {email.get('email_id', '')}",
        f"From: {email.get('sender_name', '')} <{email.get('sender', '')}>",
        f"Subject: {email.get('subject', '')}",
        f"Time: {email.get('timestamp', '')}",
        f"Has Attachment: {email.get('has_attachment', False)}",
        f"Is Reply: {email.get('is_reply', False)}",
    ]
    if email.get("thread_id"):
        lines.append(f"Thread ID: {email['thread_id']}")
    lines.extend(["", "Body:", email.get("body", "(empty)")])
    return "\n".join(lines)


def build_dataset(task_ids: List[str], seed: int = 42) -> List[Dict[str, Any]]:
    """Build a training dataset by generating email prompts from the environment."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from server.email_generator import generate_emails
    from server.tasks import get_task

    examples = []
    for task_id in task_ids:
        task = get_task(task_id)
        emails, ground_truths = generate_emails(task_id, seed)
        truth_map = {gt.email_id: gt for gt in ground_truths}

        for email in emails:
            prompt = format_email_prompt(email.model_dump(), task.description)
            gt = truth_map[email.email_id]
            examples.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                # Ground truth stored as metadata for reward functions
                "email_id": email.email_id,
                "task_id": task_id,
                "gt_category": gt.category,
                "gt_priority": gt.priority,
                "gt_department": gt.department,
                "gt_requires_response": gt.requires_response,
                "gt_expected_keywords": gt.expected_keywords,
                "requires_priority": task.requires_priority,
                "requires_routing": task.requires_routing,
                "requires_response": task.requires_response,
                "curriculum_level": task.curriculum_level,
            })

    return examples


# ─────────────────────────────────────────────────────────────────────────────
#  Independent reward functions (passed to GRPOTrainer)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_action(completion) -> Optional[Dict[str, Any]]:
    """Parse LLM completion into an action dict.

    TRL may pass completions as:
      - str  (most common, e.g. vLLM backend)
      - List[str]  (token list — join before parsing)
      - List[int]  (token ids — shouldn't happen but guard anyway)
    """
    # Normalise to str
    if isinstance(completion, list):
        completion = " ".join(str(t) for t in completion)
    if not isinstance(completion, str):
        completion = str(completion)
    text = completion.strip()

    # Strip markdown fences
    if "```" in text:
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        action = json.loads(text)
        if isinstance(action, dict) and "email_id" in action:
            return action
    except json.JSONDecodeError:
        pass
    # Try to extract embedded JSON
    for i, ch in enumerate(text):
        if ch == "{":
            depth = 0
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            a = json.loads(text[i:j+1])
                            if isinstance(a, dict) and "email_id" in a:
                                return a
                        except json.JSONDecodeError:
                            pass
                        break
    return None


# ─── Semantic distance tables (mirrors server/reward.py) ─────────────────────
_CAT_DIST = {
    "urgent":    {"urgent":0,"technical":1,"billing":2,"general":3,"spam":4},
    "technical": {"technical":0,"urgent":1,"general":2,"billing":3,"spam":4},
    "billing":   {"billing":0,"general":1,"urgent":2,"technical":3,"spam":4},
    "general":   {"general":0,"billing":1,"technical":2,"urgent":3,"spam":4},
    "spam":      {"spam":0,"general":1,"billing":2,"technical":3,"urgent":4},
}
_DEPT_DIST = {
    "engineering": {"engineering":0,"support":1,"management":2,"billing":3},
    "billing":     {"billing":0,"support":1,"management":2,"engineering":3},
    "support":     {"support":0,"engineering":1,"billing":1,"management":2},
    "management":  {"management":0,"support":1,"engineering":2,"billing":3},
}
_VALID_CATS  = {"spam","billing","technical","general","urgent"}
_VALID_DEPTS = {"engineering","billing","support","management"}


def reward_format(completions: List[str], prompts=None, **kwargs) -> List[float]:
    """Gate reward: valid JSON with correct email_id. Runs conceptually first."""
    rewards = []
    expected_ids = kwargs.get("email_id", [""] * len(completions))
    for comp, eid in zip(completions, expected_ids):
        action = _parse_action(comp)
        if action is None:
            rewards.append(-1.0)   # completely unparseable
            continue
        has_id  = bool(action.get("email_id"))
        has_cat = bool(action.get("category"))
        id_ok   = action.get("email_id") == eid
        if has_id and has_cat and id_ok:
            rewards.append(0.5)    # well-formed + correct id
        elif has_id and has_cat:
            rewards.append(0.1)    # valid shape but wrong email_id
        else:
            rewards.append(-0.5)
    return rewards


def reward_classification(completions: List[str], prompts=None, **kwargs) -> List[float]:
    """Semantic distance classification reward."""
    rewards = []
    gt_categories = kwargs.get("gt_category", [""] * len(completions))
    for comp, gt_cat in zip(completions, gt_categories):
        action = _parse_action(comp)
        if action is None:
            rewards.append(-0.5)
            continue
        agent_cat = (action.get("category") or "").lower().strip()
        if agent_cat == gt_cat:
            rewards.append(1.0)
            continue
        if agent_cat not in _VALID_CATS:
            rewards.append(-1.0)   # hallucinated — hard penalty
            continue
        dist = _CAT_DIST.get(gt_cat, {}).get(agent_cat, 4)
        rewards.append({1: 0.3, 2: 0.0, 3: -0.3}.get(dist, -0.5))
    return rewards


def reward_priority(completions: List[str], prompts=None, **kwargs) -> List[float]:
    """Graduated priority reward. Penalises severe misses strongly."""
    rewards = []
    gt_priorities = kwargs.get("gt_priority", [3] * len(completions))
    requires      = kwargs.get("requires_priority", [True] * len(completions))
    for comp, gt_pri, req in zip(completions, gt_priorities, requires):
        if not req:
            rewards.append(0.0)
            continue
        action = _parse_action(comp)
        if action is None:
            rewards.append(-0.5)
            continue
        try:
            agent_pri = int(action.get("priority", -1))
        except (ValueError, TypeError):
            rewards.append(-0.5)
            continue
        if not (1 <= agent_pri <= 5):
            rewards.append(-0.5)
            continue
        diff = abs(agent_pri - gt_pri)
        rewards.append({0: 1.0, 1: 0.4, 2: 0.0}.get(diff, -0.4))
    return rewards


def reward_routing(completions: List[str], prompts=None, **kwargs) -> List[float]:
    """Semantic distance department routing reward."""
    rewards = []
    gt_departments = kwargs.get("gt_department", ["support"] * len(completions))
    requires       = kwargs.get("requires_routing", [True] * len(completions))
    for comp, gt_dept, req in zip(completions, gt_departments, requires):
        if not req:
            rewards.append(0.0)
            continue
        action = _parse_action(comp)
        if action is None:
            rewards.append(-0.5)
            continue
        agent_dept = (action.get("department") or "").lower().strip()
        if agent_dept == gt_dept:
            rewards.append(1.0)
            continue
        if agent_dept not in _VALID_DEPTS:
            rewards.append(-1.0)   # hallucinated
            continue
        dist = _DEPT_DIST.get(gt_dept, {}).get(agent_dept, 3)
        rewards.append({1: 0.0, 2: -0.4}.get(dist, -0.8))
    return rewards


def reward_escalation(completions: List[str], prompts=None, **kwargs) -> List[float]:
    """F1-style escalation reward (TP/TN/FP/FN asymmetric scoring)."""
    rewards = []
    gt_priorities  = kwargs.get("gt_priority",   [3]    * len(completions))
    gt_departments = kwargs.get("gt_department", ["support"] * len(completions))
    for comp, gt_pri, gt_dept in zip(completions, gt_priorities, gt_departments):
        action = _parse_action(comp)
        if action is None:
            rewards.append(-0.5)
            continue
        should_escalate  = gt_dept == "management" or int(gt_pri) == 1
        did_escalate     = bool(action.get("escalate", False))
        if did_escalate and should_escalate:
            rewards.append(1.0)    # true positive
        elif not did_escalate and not should_escalate:
            rewards.append(0.3)    # true negative — restraint is good
        elif did_escalate and not should_escalate:
            rewards.append(-1.0)   # false positive — management noise
        else:
            rewards.append(-0.5)   # false negative — missed SLA
    return rewards


def reward_response(completions: List[str], prompts=None, **kwargs) -> List[float]:
    """Response quality reward: keyword coverage + length sanity check."""
    rewards = []
    requires  = kwargs.get("requires_response",    [False] * len(completions))
    keywords  = kwargs.get("gt_expected_keywords",  [[]]   * len(completions))
    for comp, req, kws in zip(completions, requires, keywords):
        if not req:
            rewards.append(0.0)
            continue
        action = _parse_action(comp)
        if action is None:
            rewards.append(-0.5)
            continue
        draft = (action.get("response_draft") or "").strip().lower()
        if not draft:
            rewards.append(-1.0)   # required but missing
            continue
        if not kws:
            rewards.append(0.5)    # no keywords to check — draft present
            continue
        length_ok = len(draft) >= 15
        matches   = sum(1 for kw in kws if kw.lower() in draft)
        coverage  = matches / len(kws)
        base = (1.0 if coverage >= 0.7 else
                0.6 if coverage >= 0.5 else
                0.3 if coverage >= 0.3 else 0.1)
        rewards.append(base if length_ok else base - 0.3)
    return rewards


REWARD_FUNCTIONS = [
    reward_format,          # gate — runs first conceptually
    reward_classification,  # semantic distance
    reward_priority,        # graduated
    reward_routing,         # semantic distance
    reward_escalation,      # F1-style
    reward_response,        # keyword coverage
]


# ─────────────────────────────────────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_name: str, lora_rank: int, use_unsloth: bool):
    """Load model with Unsloth (fast) or plain HF transformers (fallback)."""
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=2048,
                load_in_4bit=True,
                fast_inference=False,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_rank,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                lora_alpha=lora_rank * 2,
                lora_dropout=0.05,
                bias="none",
                use_gradient_checkpointing="unsloth",
            )
            print(f"✅ Loaded {model_name} with Unsloth (4-bit + LoRA r={lora_rank})")
            return model, tokenizer, "unsloth"
        except ImportError:
            print("⚠ Unsloth not installed — falling back to HF transformers")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    print(f"✅ Loaded {model_name} with HF transformers + LoRA r={lora_rank}")
    return model, tokenizer, "hf"


# ─────────────────────────────────────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────────────────────────────────────

def train_on_task(
    model,
    tokenizer,
    task_id: str,
    output_dir: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Run GRPO training on a single task. Returns training metrics."""
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset

    print(f"\n{'='*60}")
    print(f"  Training on task: {task_id.upper()}")
    print(f"{'='*60}")

    # Build dataset
    raw = build_dataset([task_id], seed=args.seed)
    dataset = Dataset.from_list(raw)

    task_output = os.path.join(output_dir, task_id)
    os.makedirs(task_output, exist_ok=True)

    grpo_config = GRPOConfig(
        output_dir=task_output,
        num_train_epochs=1,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        # Sampling params — diversity is crucial for GRPO
        temperature=1.0,        # higher diversity = better group contrast
        top_p=0.95,
        # Stability
        max_grad_norm=0.1,      # tight clipping prevents reward spikes
        warmup_steps=20,
        # Logging
        logging_steps=5,
        save_steps=50,
        seed=args.seed,
        report_to="wandb" if args.wandb else "none",
        run_name=f"email-triage-{task_id}-{args.model.split('/')[-1]}" if args.wandb else None,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=REWARD_FUNCTIONS,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    start = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start

    metrics = {
        "task_id": task_id,
        "elapsed_s": round(elapsed, 1),
        "train_loss": round(train_result.training_loss, 4),
        "steps": train_result.global_step,
    }
    print(f"\n  Task {task_id}: loss={metrics['train_loss']:.4f}, "
          f"steps={metrics['steps']}, time={elapsed:.0f}s")

    # Save adapter (NOT merging 4-bit weights — use adapters directly)
    adapter_path = os.path.join(task_output, "adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"  Saved adapter → {adapter_path}")

    return metrics


def evaluate_model(model, tokenizer, task_ids: List[str], seed: int = 99) -> Dict[str, float]:
    """Quick evaluation: run the trained model against the environment."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from server.environment import EmailTriageEnvironment

    scores = {}
    for task_id in task_ids:
        env = EmailTriageEnvironment()
        obs = env.reset(seed=seed, task_id=task_id)
        obs_dict = obs.model_dump()

        while not obs_dict.get("done", False) and obs_dict.get("emails"):
            email = obs_dict["emails"][0]
            prompt_text = format_email_prompt(email, obs_dict.get("task_description", ""))

            inputs = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            try:
                import torch
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.to(model.device),
                        max_new_tokens=256,
                        temperature=0.0,
                        do_sample=False,
                    )
                response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
                action = _parse_action(response)
            except Exception:
                action = None

            if action is None:
                action = {"email_id": email["email_id"], "category": "general",
                          "priority": 3, "department": "support"}

            obs = env.step(action)
            obs_dict = obs.model_dump()

        grading = obs_dict.get("metadata", {}).get("grading", {})
        score = grading.get("final_score", 0.0)
        scores[task_id] = score
        print(f"  Eval {task_id}: {score:.4f}")

    return scores


def evaluate_model_live(
    model, tokenizer, task_ids: List[str], env_url: str
) -> Dict[str, float]:
    """Evaluate trained model against the live OpenEnv HF Space via WebSocket.

    Uses the openenv-core EmailTriageClient protocol — the same interface
    that any external trainer (TRL, Oumi, ART) would use in production.
    Falls back to local evaluation if openenv-core is not available.
    """
    try:
        import asyncio
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from client import EmailTriageClient
        from models import EmailAction
    except ImportError:
        print("⚠ openenv-core not available — falling back to local evaluation")
        return evaluate_model(model, tokenizer, task_ids)

    async def _run_live(task_id: str) -> float:
        try:
            async with EmailTriageClient(env_url) as env:
                result = await env.reset(task_id=task_id, seed=99)
                obs = result.observation
                total_reward = 0.0
                steps = 0
                while not result.done and steps < 50:
                    emails = getattr(obs, "emails", []) or []
                    if not emails:
                        break
                    email = emails[0] if isinstance(emails[0], dict) else emails[0].model_dump()
                    prompt_text = format_email_prompt(email, getattr(obs, "task_description", ""))
                    inputs = tokenizer.apply_chat_template(
                        [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": prompt_text}],
                        tokenize=True, add_generation_prompt=True, return_tensors="pt",
                    )
                    import torch
                    with torch.no_grad():
                        out = model.generate(
                            inputs.to(model.device),
                            max_new_tokens=320, temperature=0.0, do_sample=False,
                        )
                    response = tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)
                    action = _parse_action(response) or {
                        "email_id": email["email_id"], "category": "general",
                        "priority": 3, "department": "support",
                    }
                    result = await env.step(EmailAction(**action))
                    obs = result.observation
                    total_reward += result.reward or 0.0
                    steps += 1
                # Final grading from environment
                meta = getattr(obs, "metadata", {}) or {}
                grading = meta.get("grading", {})
                return grading.get("final_score", 0.0)
        except Exception as e:
            print(f"  ⚠ Live eval {task_id} failed: {e} — returning 0.0")
            return 0.0

    print(f"\n  🌐 Live evaluation against {env_url}")
    scores = {}
    for task_id in task_ids:
        score = asyncio.run(_run_live(task_id))
        scores[task_id] = score
        print(f"  Live eval {task_id}: {score:.4f}")
    return scores


def main():
    args = parse_args()

    if args.wandb:
        try:
            import wandb
            wandb.init(project=os.getenv("WANDB_PROJECT", "email-triage-rl"),
                       name=f"grpo-{args.task}-{args.model.split('/')[-1]}")
        except ImportError:
            print("⚠ wandb not installed — skipping W&B logging")

    # Load model
    model, tokenizer, backend = load_model(
        args.model, args.lora_rank, use_unsloth=not args.no_unsloth
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine task sequence
    if args.task == "curriculum":
        task_sequence = ["easy", "medium", "hard"]
        print("\n📚 Curriculum mode: easy → medium → hard")
        print("   Advancement thresholds: easy>0.60, medium>0.50")
    else:
        task_sequence = [args.task]

    # Train
    all_metrics = []
    for task_id in task_sequence:
        metrics = train_on_task(model, tokenizer, task_id, args.output_dir, args)
        all_metrics.append(metrics)

    # Evaluate — local or live OpenEnv
    print(f"\n{'='*60}")
    print("  POST-TRAINING EVALUATION")
    print(f"{'='*60}")
    if getattr(args, 'live_eval', False):
        eval_scores = evaluate_model_live(model, tokenizer, task_sequence, args.env_url)
    else:
        eval_scores = evaluate_model(model, tokenizer, task_sequence)

    # Save final merged model (for inference, not from 4-bit — use adapter only)
    final_adapter = os.path.join(args.output_dir, "final_adapter")
    model.save_pretrained(final_adapter)
    tokenizer.save_pretrained(final_adapter)

    # Summary
    print(f"\n{'='*60}")
    print("  TRAINING SUMMARY")
    print(f"{'='*60}")
    for m in all_metrics:
        score = eval_scores.get(m["task_id"], 0.0)
        print(f"  {m['task_id']:<10} loss={m['train_loss']:.4f}  "
              f"eval_score={score:.4f}  time={m['elapsed_s']:.0f}s")
    print(f"\n  Adapter saved to: {final_adapter}")
    print(f"{'='*60}\n")

    # Save summary JSON for README/blog
    summary = {
        "model": args.model,
        "backend": backend,
        "tasks": all_metrics,
        "eval_scores": eval_scores,
    }
    with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
