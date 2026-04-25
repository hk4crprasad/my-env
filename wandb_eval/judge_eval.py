"""
judge_eval.py — W&B Evaluation Suite for Email Triage RL Environment
======================================================================
OpenEnv Hackathon 2026 · Team Ctrl-Alt-Defeat

PURPOSE
-------
Gives judges a single runnable script that:
  1. Validates the environment (all 3 tasks, deterministic grading)
  2. Measures reward component separation (perfect vs random vs adversarial actions)
  3. Benchmarks a baseline LLM (any OpenAI-compatible API, free via HF Router)
  4. Benchmarks the trained GRPO adapter for comparison
  5. Logs all results to a public Weights & Biases run with panels, tables, and plots

All W&B runs are logged to:
  wandb.ai/ctrl-alt-defeat/email-triage-eval   (or your own project — see --wandb-project)

USAGE
-----
# 1. Minimum — just reward analysis + environment validation (no LLM, no GPU needed):
    pip install wandb
    python wandb_eval/judge_eval.py --mode env

# 2. Full baseline + trained comparison (needs HF token):
    export HF_TOKEN="hf_..."
    python wandb_eval/judge_eval.py --mode full

# 3. With local trained adapter (needs GPU):
    export HF_TOKEN="hf_..."
    python wandb_eval/judge_eval.py --mode full --use-local-model

# 4. Custom LLM endpoint:
    python wandb_eval/judge_eval.py --mode full \\
        --api-base https://api.openai.com/v1 \\
        --model gpt-4o-mini

ENVIRONMENT VARIABLES (alternatives to CLI flags)
--------------------------------------------------
    HF_TOKEN          Hugging Face token (for inference + trained adapter)
    API_BASE_URL      LLM API endpoint   (default: HF Inference Router)
    MODEL_NAME        LLM model ID       (default: openai/gpt-oss-120b)
    WANDB_PROJECT     W&B project name   (default: email-triage-eval)
    WANDB_ENTITY      W&B entity/team    (default: your W&B username)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# ── path setup so we can import the environment directly ──────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# ─────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="W&B evaluation suite — Email Triage RL Environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--mode", choices=["env", "baseline", "full"], default="full",
        help=(
            "env      → environment validation + reward analysis only (no LLM)\n"
            "baseline → env + baseline LLM scoring\n"
            "full     → env + baseline + trained model (default)"
        ),
    )
    p.add_argument("--seed", type=int, default=42, help="Evaluation seed")
    p.add_argument(
        "--api-base", default=None,
        help="LLM API base URL (default: HF Inference Router)",
    )
    p.add_argument("--model", default=None, help="LLM model name")
    p.add_argument("--api-key", default=None, help="API key / HF token")
    p.add_argument(
        "--use-local-model", action="store_true",
        help="Load trained LoRA adapter locally (requires GPU + transformers + peft)",
    )
    p.add_argument(
        "--adapter-id", default="Hk4crprasad/email-triage-grpo",
        help="HF Hub adapter ID for trained model",
    )
    p.add_argument(
        "--base-model", default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model for the adapter",
    )
    p.add_argument(
        "--wandb-project", default=None,
        help="W&B project name (default: email-triage-eval)",
    )
    p.add_argument(
        "--wandb-entity", default=None,
        help="W&B entity/team (default: current W&B user)",
    )
    p.add_argument(
        "--no-wandb", action="store_true",
        help="Disable W&B logging (print results to stdout only)",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────
#  Imports (lazy — so the script starts fast even without optional deps)
# ─────────────────────────────────────────────────────────────────────────

def _import_env():
    from server.environment import EmailTriageEnvironment
    from server.email_generator import generate_emails
    from server.reward import (
        reward_classification, reward_priority, reward_routing,
        reward_response_quality, reward_escalation, reward_format_compliance,
        REWARD_RUBRIC,
    )
    from server.tasks import TASKS
    return (
        EmailTriageEnvironment, generate_emails,
        reward_classification, reward_priority, reward_routing,
        reward_response_quality, reward_escalation, reward_format_compliance,
        REWARD_RUBRIC, TASKS,
    )


# ─────────────────────────────────────────────────────────────────────────
#  Section 1 — Environment validation
# ─────────────────────────────────────────────────────────────────────────

def run_env_validation(seed: int) -> Dict[str, Any]:
    """
    Validates reset/step/grading for all 3 tasks with perfect actions.
    Returns a dict of pass/fail counts and final scores.
    """
    print("\n" + "─" * 60)
    print("  SECTION 1: Environment Validation")
    print("─" * 60)

    (EmailTriageEnvironment, generate_emails,
     rc, rp, rr, rq, re, rf, REWARD_RUBRIC, TASKS) = _import_env()

    results = {}
    for task_id in ["easy", "medium", "hard"]:
        env = EmailTriageEnvironment()
        obs = env.reset(seed=seed, task_id=task_id)
        obs_d = obs.model_dump()

        assert not obs_d["done"], f"{task_id}: done=True immediately after reset"
        assert obs_d["emails"], f"{task_id}: no emails after reset"
        assert obs_d["steps_remaining"] > 0, f"{task_id}: steps_remaining=0"

        emails, gts = generate_emails(task_id, seed)
        truth_map = {gt.email_id: gt for gt in gts}

        step_rewards: List[float] = []
        steps = 0
        while not obs_d["done"] and obs_d.get("emails"):
            email = obs_d["emails"][0]
            eid = email["email_id"]
            gt = truth_map[eid]
            # perfect action
            action = {
                "email_id": eid,
                "category": gt.category,
                "priority": gt.priority,
                "department": gt.department,
                "response_draft": " ".join(gt.expected_keywords) if gt.expected_keywords else None,
                "escalate": gt.department == "management" or gt.priority == 1,
            }
            obs = env.step(action)
            obs_d = obs.model_dump()
            step_rewards.append(obs_d["step_reward"])
            steps += 1

        grading = obs_d.get("metadata", {}).get("grading", {})
        final_score = grading.get("final_score", 0.0)
        dim_scores = grading.get("dimension_scores", {})

        results[task_id] = {
            "final_score": final_score,
            "steps_used": steps,
            "avg_step_reward": sum(step_rewards) / max(len(step_rewards), 1),
            "dimension_scores": dim_scores,
            "pass": final_score > 0.0,
        }
        status = "✓ PASS" if results[task_id]["pass"] else "✗ FAIL"
        print(f"  [{status}] {task_id:<8} score={final_score:.4f}  steps={steps}  "
              f"avg_reward={results[task_id]['avg_step_reward']:+.3f}")

    all_pass = all(v["pass"] for v in results.values())
    print(f"\n  {'All 3 tasks PASS ✅' if all_pass else 'Some tasks FAILED ❌'}")
    return results


# ─────────────────────────────────────────────────────────────────────────
#  Section 2 — Reward component analysis
# ─────────────────────────────────────────────────────────────────────────

COMPONENTS = ["classification", "priority", "routing", "response", "escalation", "format"]


def _score_one_email(email_dict, gt, action, valid_ids, rc, rp, rr, rq, re, rf):
    return {
        "classification": rc(action, gt),
        "priority": rp(action, gt),
        "routing": rr(action, gt),
        "response": rq(action, gt),
        "escalation": re(action, gt),
        "format": rf(action, valid_ids),
    }


def run_reward_analysis(seed: int) -> Dict[str, Any]:
    """
    Scores every hard-task email with 3 strategies:
      - perfect: correct action for every field
      - random:  random valid values
      - adversarial: always choose the most common wrong answer

    Returns per-component mean rewards for all 3 strategies.
    """
    print("\n" + "─" * 60)
    print("  SECTION 2: Reward Component Analysis")
    print("─" * 60)

    (_, generate_emails,
     rc, rp, rr, rq, re, rf, REWARD_RUBRIC, TASKS) = _import_env()

    emails, gts = generate_emails("hard", seed)
    truth_map = {gt.email_id: gt for gt in gts}
    valid_ids = {e.email_id for e in emails}

    rng = random.Random(seed)
    VALID_CATS  = ["spam", "billing", "technical", "general", "urgent"]
    VALID_DEPTS = ["engineering", "billing", "support", "management"]

    strategy_scores: Dict[str, Dict[str, List[float]]] = {
        "perfect":     {c: [] for c in COMPONENTS},
        "random":      {c: [] for c in COMPONENTS},
        "adversarial": {c: [] for c in COMPONENTS},
    }

    per_email_rows = []

    for email in emails:
        gt = truth_map[email.email_id]

        # Perfect action
        perfect = {
            "email_id": email.email_id,
            "category": gt.category,
            "priority": gt.priority,
            "department": gt.department,
            "response_draft": " ".join(gt.expected_keywords) if gt.expected_keywords else None,
            "escalate": gt.department == "management" or gt.priority == 1,
        }

        # Random action (valid values only)
        rand_cat  = rng.choice(VALID_CATS)
        rand_dept = rng.choice(VALID_DEPTS)
        rand_pri  = rng.randint(1, 5)
        random_action = {
            "email_id": email.email_id,
            "category": rand_cat,
            "priority": rand_pri,
            "department": rand_dept,
            "response_draft": "thank you for reaching out",
            "escalate": rng.choice([True, False]),
        }

        # Adversarial: always pick the WRONG valid answer
        adv_cat  = next(c for c in VALID_CATS  if c != gt.category)
        adv_dept = next(d for d in VALID_DEPTS if d != gt.department)
        adv_pri  = (gt.priority % 5) + 1  # shift priority by 1 cyclically
        adversarial = {
            "email_id": email.email_id,
            "category": adv_cat,
            "priority": adv_pri,
            "department": adv_dept,
            "response_draft": None,
            "escalate": not (gt.department == "management" or gt.priority == 1),
        }

        for strategy, action in [
            ("perfect", perfect), ("random", random_action), ("adversarial", adversarial)
        ]:
            scores = _score_one_email(email.model_dump(), gt, action, valid_ids,
                                      rc, rp, rr, rq, re, rf)
            for comp in COMPONENTS:
                strategy_scores[strategy][comp].append(scores[comp])

        # Row for W&B table
        p_scores = _score_one_email(email.model_dump(), gt, perfect, valid_ids,
                                    rc, rp, rr, rq, re, rf)
        r_scores = _score_one_email(email.model_dump(), gt, random_action, valid_ids,
                                    rc, rp, rr, rq, re, rf)
        per_email_rows.append({
            "email_id":    email.email_id,
            "subject":     email.subject[:60],
            "true_cat":    gt.category,
            "true_pri":    gt.priority,
            "true_dept":   gt.department,
            "perfect_total": round(sum(p_scores.values()), 4),
            "random_total":  round(sum(r_scores.values()), 4),
            **{f"perfect_{c}": round(p_scores[c], 4) for c in COMPONENTS},
            **{f"random_{c}":  round(r_scores[c], 4) for c in COMPONENTS},
        })

    # Compute means
    means: Dict[str, Dict[str, float]] = {}
    for strategy in ["perfect", "random", "adversarial"]:
        means[strategy] = {
            c: round(sum(strategy_scores[strategy][c]) / len(emails), 4)
            for c in COMPONENTS
        }

    # Print table
    print(f"\n  {'Component':<16} {'Perfect':>9} {'Random':>9} {'Adversarial':>12}  {'Max':>7}")
    print(f"  {'─'*16} {'─'*9} {'─'*9} {'─'*12}  {'─'*7}")
    maxes = {"classification": 0.20, "priority": 0.15, "routing": 0.15,
             "response": 0.30, "escalation": 0.05, "format": 0.05}
    for c in COMPONENTS:
        mx = maxes.get(c, "–")
        print(f"  {c:<16} "
              f"{means['perfect'][c]:>+9.3f} "
              f"{means['random'][c]:>+9.3f} "
              f"{means['adversarial'][c]:>+12.3f}  "
              f"{mx if isinstance(mx, str) else f'+{mx:.2f}':>7}")

    return {
        "means": means,
        "per_email_rows": per_email_rows,
        "reward_rubric": {c: {"max": maxes.get(c, 0)} for c in COMPONENTS},
    }


# ─────────────────────────────────────────────────────────────────────────
#  Section 3 — LLM agent scoring
# ─────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert email triage agent. For each email, respond with a single valid JSON object (no markdown):
{
  "email_id": "<id>",
  "category": "<spam|billing|technical|general|urgent>",
  "priority": <1-5 where 1=critical, 5=low>,
  "department": "<engineering|billing|support|management>",
  "response_draft": "<reply text or null>",
  "escalate": <true|false>
}

spam=unsolicited/phishing, billing=payments/invoices, technical=bugs/API,
general=inquiries/requests, urgent=outages/security/legal

Priorities: 1=critical(system down/breach), 2=high, 3=medium, 4=low, 5=lowest(spam)
Departments: engineering=bugs/API/security, billing=payments, support=general/spam, management=legal/compliance

Watch for PHISHING (suspicious domains, bit.ly links) and RED HERRINGS (subject says URGENT but body says TEST).
Respond with ONLY the JSON. No other text."""


def _parse_action(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if "```" in text:
        text = "\n".join(l for l in text.split("\n") if not l.strip().startswith("```")).strip()
    try:
        a = json.loads(text)
        if isinstance(a, dict) and "email_id" in a:
            return a
    except json.JSONDecodeError:
        pass
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


def _format_prompt(email: Dict[str, Any], task_desc: str) -> str:
    lines = [
        f"TASK: {task_desc[:200]}",
        f"Email ID: {email['email_id']}",
        f"From: {email.get('sender_name', '')} <{email.get('sender', '')}>",
        f"Subject: {email.get('subject', '')}",
        f"Has Attachment: {email.get('has_attachment', False)}",
        f"Is Reply: {email.get('is_reply', False)}",
    ]
    if email.get("thread_id"):
        lines.append(f"Thread ID: {email['thread_id']}")
    lines += ["", "Body:", email.get("body", "")]
    return "\n".join(lines)


def run_llm_eval(
    label: str,
    seed: int,
    client=None,
    model_name: str = "",
    local_model=None,
    local_tokenizer=None,
) -> Dict[str, Any]:
    """
    Runs an LLM agent against all 3 tasks. Returns episode scores + per-step data.
    Works in API mode (client) or local adapter mode (local_model/tokenizer).
    """
    print(f"\n  ── {label} ──")

    (EmailTriageEnvironment, _, *_rest) = _import_env()

    results: Dict[str, Any] = {}
    all_steps: List[Dict] = []

    for task_id in ["easy", "medium", "hard"]:
        env = EmailTriageEnvironment()
        obs = env.reset(seed=seed, task_id=task_id)
        obs_d = obs.model_dump()

        step_data: List[Dict] = []
        step = 0
        max_steps = {"easy": 10, "medium": 25, "hard": 40}[task_id]

        while not obs_d["done"] and obs_d.get("emails") and step < max_steps:
            email = obs_d["emails"][0]
            prompt = _format_prompt(email, obs_d.get("task_description", ""))

            # Get completion
            action: Optional[Dict] = None
            latency_ms = 0
            try:
                t0 = time.time()
                if local_model is not None:
                    import torch
                    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user",   "content": prompt}]
                    ids = local_tokenizer.apply_chat_template(
                        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                    ).to(next(local_model.parameters()).device)
                    with torch.no_grad():
                        out = local_model.generate(ids, max_new_tokens=512,
                                                   do_sample=False, temperature=None,
                                                   pad_token_id=local_tokenizer.pad_token_id)
                    raw = local_tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
                elif client is not None:
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                                  {"role": "user",   "content": prompt}],
                        temperature=0.0, max_tokens=512,
                    )
                    raw = resp.choices[0].message.content or ""
                else:
                    raw = ""
                latency_ms = int((time.time() - t0) * 1000)
                action = _parse_action(raw)
            except Exception as e:
                print(f"    ⚠ step {step}: {e}")

            if action is None:
                action = {
                    "email_id": email["email_id"],
                    "category": "general",
                    "priority": 3,
                    "department": "support",
                    "response_draft": None,
                    "escalate": False,
                }

            obs = env.step(action)
            obs_d = obs.model_dump()
            step += 1

            step_data.append({
                "task_id":    task_id,
                "step":       step,
                "email_id":   email["email_id"],
                "subject":    email.get("subject", "")[:60],
                "pred_cat":   action.get("category", ""),
                "pred_pri":   action.get("priority", ""),
                "pred_dept":  action.get("department", ""),
                "step_reward": obs_d.get("step_reward", 0.0),
                "feedback":   (obs_d.get("action_feedback") or "")[:120],
                "latency_ms": latency_ms,
                "cumulative_reward": obs_d.get("cumulative_reward", 0.0),
            })

        grading = obs_d.get("metadata", {}).get("grading", {})
        final_score = grading.get("final_score", 0.0)
        dim_scores  = grading.get("dimension_scores", {})

        results[task_id] = {
            "final_score": final_score,
            "dimension_scores": dim_scores,
            "steps_used": step,
            "emails_processed": grading.get("emails_processed", 0),
            "emails_total": grading.get("emails_total", 0),
        }
        all_steps.extend(step_data)

        print(f"    {task_id:<8} score={final_score:.4f}  "
              f"dims={json.dumps({k: round(v, 2) for k, v in dim_scores.items()})}")

    results["_steps"] = all_steps
    return results


# ─────────────────────────────────────────────────────────────────────────
#  Section 4 — W&B logging
# ─────────────────────────────────────────────────────────────────────────

def log_to_wandb(
    env_results: Dict,
    reward_analysis: Dict,
    baseline_results: Optional[Dict],
    trained_results: Optional[Dict],
    args: argparse.Namespace,
):
    try:
        import wandb
    except ImportError:
        print("\n⚠ wandb not installed — skipping W&B logging.")
        print("  Install: pip install wandb")
        return

    project = args.wandb_project or os.getenv("WANDB_PROJECT", "email-triage-eval")
    entity  = args.wandb_entity  or os.getenv("WANDB_ENTITY",  None)

    run = wandb.init(
        project=project,
        entity=entity,
        name=f"judge-eval-{args.mode}-seed{args.seed}",
        tags=["email-triage", "openenv", "ctrl-alt-defeat", "judge-eval"],
        config={
            "mode":       args.mode,
            "seed":       args.seed,
            "model":      args.model or "openai/gpt-oss-120b",
            "adapter_id": args.adapter_id,
            "hf_space":   "https://huggingface.co/spaces/Hk4crprasad/email-triage-env",
            "github":     "https://github.com/hk4crprasad/my-env",
        },
    )

    # ── 1. Environment validation results ─────────────────────────────────
    val_table = wandb.Table(
        columns=["task", "final_score", "steps_used", "avg_step_reward", "pass"],
    )
    for task_id, r in env_results.items():
        val_table.add_data(
            task_id, r["final_score"], r["steps_used"],
            round(r["avg_step_reward"], 4), "✅" if r["pass"] else "❌",
        )
    wandb.log({"validation/results_table": val_table})

    for task_id, r in env_results.items():
        wandb.log({
            f"validation/{task_id}/final_score": r["final_score"],
            f"validation/{task_id}/avg_step_reward": r["avg_step_reward"],
        })

    # ── 2. Reward component analysis ──────────────────────────────────────
    means = reward_analysis["means"]
    rubric_table = wandb.Table(
        columns=["component", "perfect", "random", "adversarial", "max_possible", "separation"],
    )
    max_map = {"classification": 0.20, "priority": 0.15, "routing": 0.15,
               "response": 0.30, "escalation": 0.05, "format": 0.05}
    for c in COMPONENTS:
        mx = max_map.get(c, 0.05)
        separation = round(means["perfect"][c] - means["random"][c], 4)
        rubric_table.add_data(
            c,
            means["perfect"][c], means["random"][c], means["adversarial"][c],
            mx, separation,
        )
    wandb.log({"reward_analysis/component_table": rubric_table})

    # Bar chart data — log each strategy's component means
    for strategy in ["perfect", "random", "adversarial"]:
        for c in COMPONENTS:
            wandb.log({f"reward_analysis/{strategy}/{c}": means[strategy][c]})

    # Per-email table
    per_email_cols = list(reward_analysis["per_email_rows"][0].keys())
    per_email_table = wandb.Table(columns=per_email_cols)
    for row in reward_analysis["per_email_rows"]:
        per_email_table.add_data(*[row[c] for c in per_email_cols])
    wandb.log({"reward_analysis/per_email_table": per_email_table})

    # ── 3. Baseline vs trained comparison ─────────────────────────────────
    tasks = ["easy", "medium", "hard"]

    if baseline_results:
        for task_id in tasks:
            r = baseline_results.get(task_id, {})
            wandb.log({
                f"baseline/{task_id}/final_score": r.get("final_score", 0.0),
                **{f"baseline/{task_id}/{k}": v
                   for k, v in r.get("dimension_scores", {}).items()},
            })

        # Per-step table
        if baseline_results.get("_steps"):
            step_cols = list(baseline_results["_steps"][0].keys())
            step_table = wandb.Table(columns=step_cols)
            for row in baseline_results["_steps"]:
                step_table.add_data(*[row[c] for c in step_cols])
            wandb.log({"baseline/step_details": step_table})

    if trained_results:
        for task_id in tasks:
            r = trained_results.get(task_id, {})
            wandb.log({
                f"trained/{task_id}/final_score": r.get("final_score", 0.0),
                **{f"trained/{task_id}/{k}": v
                   for k, v in r.get("dimension_scores", {}).items()},
            })

        if trained_results.get("_steps"):
            step_cols = list(trained_results["_steps"][0].keys())
            step_table = wandb.Table(columns=step_cols)
            for row in trained_results["_steps"]:
                step_table.add_data(*[row[c] for c in step_cols])
            wandb.log({"trained/step_details": step_table})

    # ── 4. Improvement summary table ──────────────────────────────────────
    if baseline_results and trained_results:
        impr_table = wandb.Table(
            columns=["task", "baseline", "trained", "delta", "delta_pct"],
        )
        for task_id in tasks:
            b = baseline_results.get(task_id, {}).get("final_score", 0.0)
            t = trained_results.get(task_id, {}).get("final_score", 0.0)
            delta = round(t - b, 4)
            delta_pct = round((delta / max(b, 0.001)) * 100, 1)
            impr_table.add_data(task_id, round(b, 4), round(t, 4), delta, delta_pct)
            wandb.log({
                f"improvement/{task_id}/delta": delta,
                f"improvement/{task_id}/baseline": b,
                f"improvement/{task_id}/trained": t,
            })
        wandb.log({"improvement/summary_table": impr_table})

    # ── 5. Upload committed plots ──────────────────────────────────────────
    plots_dir = os.path.join(_ROOT, "plots")
    for fname in ["reward_spread.png", "score_comparison.png",
                  "dimension_breakdown.png", "training_curve.png"]:
        fpath = os.path.join(plots_dir, fname)
        if os.path.exists(fpath):
            key = fname.replace(".png", "").replace("_", "/")
            wandb.log({f"plots/{key}": wandb.Image(fpath)})
            print(f"  ↑ Uploaded plot: {fname}")

    run_url = run.get_url()
    run.finish()
    print(f"\n  ✅ W&B run complete: {run_url}")
    return run_url


# ─────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    api_key  = args.api_key or os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    api_base = args.api_base or os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
    model    = args.model or os.getenv("MODEL_NAME") or "openai/gpt-oss-120b"

    print("=" * 60)
    print("  EMAIL TRIAGE RL — JUDGE EVALUATION SUITE")
    print(f"  Mode    : {args.mode}")
    print(f"  Seed    : {args.seed}")
    if args.mode != "env":
        print(f"  Model   : {model}")
        print(f"  API     : {api_base}")
    print("=" * 60)

    # ── Section 1: Environment validation ─────────────────────────────────
    env_results = run_env_validation(args.seed)

    # ── Section 2: Reward analysis ────────────────────────────────────────
    reward_analysis = run_reward_analysis(args.seed)

    baseline_results = None
    trained_results  = None

    # ── Section 3: Baseline LLM ───────────────────────────────────────────
    if args.mode in ("baseline", "full"):
        print("\n" + "─" * 60)
        print("  SECTION 3: Baseline LLM Evaluation")
        print("─" * 60)
        if not api_key:
            print("  ⚠ No API key found — set HF_TOKEN or use --api-key")
            print("  Skipping LLM sections.")
        else:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=api_base)
            baseline_results = run_llm_eval(
                label=f"Baseline ({model})",
                seed=args.seed,
                client=client,
                model_name=model,
            )

    # ── Section 4: Trained adapter ────────────────────────────────────────
    if args.mode == "full" and baseline_results is not None:
        print("\n" + "─" * 60)
        print("  SECTION 4: Trained Adapter Evaluation")
        print("─" * 60)

        if args.use_local_model:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from peft import PeftModel
                import torch
                print(f"  Loading base: {args.base_model}")
                tok = AutoTokenizer.from_pretrained(args.base_model, token=api_key)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                base = AutoModelForCausalLM.from_pretrained(
                    args.base_model,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto", token=api_key,
                )
                print(f"  Loading adapter: {args.adapter_id}")
                model_obj = PeftModel.from_pretrained(base, args.adapter_id, token=api_key)
                model_obj.eval()
                print(f"  ✅ Loaded. Device: {next(model_obj.parameters()).device}")
                trained_results = run_llm_eval(
                    label=f"Trained ({args.adapter_id})",
                    seed=args.seed,
                    local_model=model_obj,
                    local_tokenizer=tok,
                )
            except Exception as e:
                print(f"  ⚠ Could not load adapter: {e}")
                print("  Tip: pip install transformers peft accelerate")
        else:
            # API mode: use the adapter served via HF Inference or a compatible endpoint
            adapter_model = f"{args.adapter_id}"
            print(f"  Using adapter via API: {adapter_model}")
            print(f"  (Set --use-local-model to load weights directly)")
            try:
                from openai import OpenAI
                trained_client = OpenAI(api_key=api_key, base_url=api_base)
                trained_results = run_llm_eval(
                    label=f"Trained adapter ({adapter_model})",
                    seed=args.seed,
                    client=trained_client,
                    model_name=adapter_model,
                )
            except Exception as e:
                print(f"  ⚠ Adapter API call failed: {e}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"\n  {'Task':<10} {'Env (perfect)':>14}", end="")
    if baseline_results:
        print(f"  {'Baseline':>10}", end="")
    if trained_results:
        print(f"  {'Trained':>10}  {'Δ':>6}", end="")
    print()
    print(f"  {'─'*10} {'─'*14}", end="")
    if baseline_results:
        print(f"  {'─'*10}", end="")
    if trained_results:
        print(f"  {'─'*10}  {'─'*6}", end="")
    print()

    for task_id in ["easy", "medium", "hard"]:
        env_score = env_results.get(task_id, {}).get("final_score", 0.0)
        print(f"  {task_id:<10} {env_score:>14.4f}", end="")
        if baseline_results:
            b = baseline_results.get(task_id, {}).get("final_score", 0.0)
            print(f"  {b:>10.4f}", end="")
        if trained_results:
            t = trained_results.get(task_id, {}).get("final_score", 0.0)
            b = (baseline_results or {}).get(task_id, {}).get("final_score", 0.0)
            delta = t - b if baseline_results else 0.0
            print(f"  {t:>10.4f}  {delta:>+6.4f}", end="")
        print()

    # ── W&B upload ────────────────────────────────────────────────────────
    if not args.no_wandb:
        print("\n" + "─" * 60)
        print("  SECTION 5: Uploading to W&B")
        print("─" * 60)
        log_to_wandb(env_results, reward_analysis, baseline_results, trained_results, args)
    else:
        print("\n  (W&B logging disabled — pass --no-wandb=False to enable)")


if __name__ == "__main__":
    main()
