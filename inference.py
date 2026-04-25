"""
inference.py — Email Triage RL Environment
============================================
OpenEnv Hackathon 2026 · Team Ctrl-Alt-Defeat

PURPOSE
-------
Runs an LLM agent against all 3 triage tasks (easy / medium / hard) and
prints structured output in the exact format the hackathon validator expects:

    [START] task=easy env=email_triage model=<model>
    [STEP]  step=1 action={...} reward=0.35 done=false error=null
    [END]   success=true steps=5 score=0.9200 rewards=0.35,0.20,...

HOW TO RUN
----------
Mode 1 — API (default, uses HF Inference Router or any OpenAI-compatible API):

    export HF_TOKEN="hf_..."
    export MODEL_NAME="openai/gpt-oss-120b"   # default — free on HF router
    python inference.py

Mode 2 — Trained adapter (our GRPO fine-tuned model, no API needed):

    pip install transformers peft accelerate
    export HF_TOKEN="hf_..."
    export USE_LOCAL_MODEL=1
    python inference.py

    This loads Hk4crprasad/email-triage-grpo (43 MB LoRA adapter) on top of
    Qwen/Qwen2.5-3B-Instruct and runs it locally.

ENVIRONMENT VARIABLES
---------------------
  HF_TOKEN         Your Hugging Face token (also accepted as API_KEY)
  API_BASE_URL     LLM API endpoint (default: https://router.huggingface.co/v1)
  MODEL_NAME       Model identifier (default: openai/gpt-oss-120b)
  USE_LOCAL_MODEL  Set to 1 to use the trained LoRA adapter (default: 0)
  ADAPTER_MODEL_ID Adapter to load (default: Hk4crprasad/email-triage-grpo)
  BASE_MODEL_ID    Base model for adapter (default: Qwen/Qwen2.5-3B-Instruct)

LINKS
-----
  Environment (HF Space): https://huggingface.co/spaces/Hk4crprasad/email-triage-env
  Trained adapter:        https://huggingface.co/Hk4crprasad/email-triage-grpo
  Source code (GitHub):   https://github.com/hk4crprasad/my-env
"""

import json
import os
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────

API_BASE_URL = (
    os.getenv("API_BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
    or "https://router.huggingface.co/v1"
)
API_KEY = (
    os.getenv("HF_TOKEN")
    or os.getenv("API_KEY")
    or os.getenv("OPENAI_API_KEY")
)
MODEL_NAME = (
    os.getenv("MODEL_NAME")
    or os.getenv("OPENAI_CHAT_MODELS")
    or "openai/gpt-oss-120b"
)

# ── Trained adapter (LoRA) uploaded to HF after GRPO training ──────────────
# Base model the adapter was trained on
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
# The fine-tuned LoRA adapter (safetensors on HF Hub)
ADAPTER_MODEL_ID = os.getenv("ADAPTER_MODEL_ID", "Hk4crprasad/email-triage-grpo")
# Set USE_LOCAL_MODEL=1 to load the adapter locally instead of hitting an API
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "0") == "1"
# 4-bit quantisation — auto-enabled on GPUs with <6 GB VRAM to fit Qwen2.5-3B.
# Override with LOAD_IN_4BIT=0 to disable (requires >=6 GB VRAM).
_LOAD_IN_4BIT_ENV = os.getenv("LOAD_IN_4BIT", "auto")

BENCHMARK = "email_triage"
MAX_STEPS_PER_TASK = {"easy": 10, "medium": 25, "hard": 40}
TEMPERATURE = 0.0  # Deterministic for reproducibility
MAX_TOKENS = 1024

# ─────────────────────────────────────────────────────────────────────────
#  System prompt
# ─────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert email triage agent. You receive emails and must process each one
by providing a JSON action.

For EACH email, respond with a single valid JSON object (no markdown, no explanation):
{
    "email_id": "<the email's ID>",
    "category": "<spam|billing|technical|general|urgent>",
    "priority": <1-5 where 1=critical, 5=low>,
    "department": "<engineering|billing|support|management>",
    "response_draft": "<draft reply if the email is urgent/critical, else null>",
    "escalate": <true if needs management attention, false otherwise>
}

Classification guidelines:
- spam: Unsolicited marketing, phishing, scam emails. Look for suspicious sender domains,
  too-good-to-be-true offers, fake urgency, and unsubscribe links.
- billing: Payment issues, invoices, charges, refunds, subscription changes.
- technical: Bug reports, API issues, feature not working, technical questions.
- general: General inquiries, feature requests, partnerships, onboarding, non-specific.
- urgent: Critical system outages, security vulnerabilities, legal threats, data breaches.

Priority guidelines:
- 1 (Critical): System down, security breach, legal action, major customer threatening to leave
- 2 (High): Service degradation, account lockout, multi-issue complaints
- 3 (Medium): API docs missing, billing discrepancy, data export questions
- 4 (Low): Feature requests, general inquiries, newsletters
- 5 (Lowest): Spam, automated receipts, test alerts

Department guidelines:
- engineering: Technical bugs, API issues, security vulnerabilities, system access setup
- billing: Payment issues, invoice disputes, refunds, subscription changes
- support: General inquiries, account help, spam handling, multi-language support
- management: Legal threats, partnership proposals, escalations, compliance audits

Important:
- Watch for PHISHING: emails that look urgent but have suspicious sender domains (misspelled, .net instead of .com)
- Watch for RED HERRINGS: Subject says "URGENT" or "CRITICAL" but body reveals it's a test, marketing, or low-priority
- For THREAD emails (is_reply=true with thread_id): consider the context from related emails
- Draft responses ONLY for truly critical emails (priority 1 with urgent/technical category)

Respond with ONLY the JSON object. No other text.
""").strip()


# ─────────────────────────────────────────────────────────────────────────
#  Environment interaction
# ─────────────────────────────────────────────────────────────────────────

def format_emails_for_prompt(observation: Dict[str, Any]) -> str:
    """Format the observation into a human-readable prompt for the LLM."""
    emails = observation.get("emails", [])
    task_desc = observation.get("task_description", "")
    stats = observation.get("inbox_stats", {})
    steps_remaining = observation.get("steps_remaining", "?")

    lines = [
        f"=== TASK: {observation.get('task_id', 'unknown').upper()} ===",
        task_desc,
        "",
        f"Inbox: {stats.get('unprocessed', 0)} unprocessed / {stats.get('total', 0)} total",
        f"Steps remaining: {steps_remaining}",
        "",
    ]

    if not emails:
        lines.append("No unprocessed emails remaining.")
        return "\n".join(lines)

    # Show the FIRST unprocessed email (process one at a time)
    email = emails[0]
    lines.append("--- NEXT EMAIL TO PROCESS ---")
    lines.append(f"Email ID: {email.get('email_id', 'unknown')}")
    lines.append(f"From: {email.get('sender_name', '')} <{email.get('sender', '')}>")
    lines.append(f"Subject: {email.get('subject', '')}")
    lines.append(f"Time: {email.get('timestamp', '')}")
    lines.append(f"Has Attachment: {email.get('has_attachment', False)}")
    lines.append(f"Is Reply: {email.get('is_reply', False)}")
    if email.get("thread_id"):
        lines.append(f"Thread ID: {email['thread_id']}")
    lines.append("")
    lines.append("Body:")
    lines.append(email.get("body", "(empty)"))
    lines.append("--- END EMAIL ---")

    return "\n".join(lines)


def parse_llm_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Parse the LLM's JSON response into an action dict."""
    text = response_text.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        action = json.loads(text)
        if isinstance(action, dict) and "email_id" in action:
            return action
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from mixed text
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
                            action = json.loads(text[i : j + 1])
                            if isinstance(action, dict) and "email_id" in action:
                                return action
                        except json.JSONDecodeError:
                            pass
                        break
    return None


def make_fallback_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    """Create a fallback action when LLM fails to respond properly."""
    emails = observation.get("emails", [])
    if not emails:
        return {"email_id": "unknown", "category": "general", "priority": 3, "department": "support"}
    email = emails[0]
    return {
        "email_id": email.get("email_id", "unknown"),
        "category": "general",
        "priority": 3,
        "department": "support",
        "response_draft": None,
        "escalate": False,
    }


# ─────────────────────────────────────────────────────────────────────────
#  Local adapter inference (trained LoRA from Hk4crprasad/email-triage-grpo)
# ─────────────────────────────────────────────────────────────────────────

def load_local_adapter():
    """
    Load the trained LoRA adapter on top of the base model.
    Returns (model, tokenizer) ready for generation.

    Adapter: Hk4crprasad/email-triage-grpo  (43 MB safetensors)
    Base:    Qwen/Qwen2.5-3B-Instruct

    4-bit quantisation (BitsAndBytes) is auto-enabled when VRAM < 6 GB
    so the 3B model fits in 4 GB cards (e.g. RTX 3050 Laptop).
    Override: LOAD_IN_4BIT=0  to force float16 (needs >=6 GB VRAM)
              LOAD_IN_4BIT=1  to force 4-bit regardless
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        import torch
    except ImportError:
        raise ImportError(
            "Local adapter mode requires: pip install transformers peft accelerate bitsandbytes"
        )

    hf_token = API_KEY or None

    # ── Decide quantisation ───────────────────────────────────────────────
    use_4bit = False
    if _LOAD_IN_4BIT_ENV == "1":
        use_4bit = True
    elif _LOAD_IN_4BIT_ENV == "0":
        use_4bit = False
    else:
        # auto: enable 4-bit when GPU VRAM < 6 GB
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            use_4bit = vram_gb < 6.0
            print(f"ℹ  GPU VRAM: {vram_gb:.1f} GB → 4-bit={'ON' if use_4bit else 'OFF'}", flush=True)

    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("🔧 Using 4-bit NF4 quantisation (BitsAndBytes)", flush=True)

    print(f"🔄 Loading base model: {BASE_MODEL_ID}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        token=hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=torch.float16 if (torch.cuda.is_available() and not use_4bit) else None,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
    )

    print(f"🔄 Loading LoRA adapter: {ADAPTER_MODEL_ID}", flush=True)
    model = PeftModel.from_pretrained(
        base,
        ADAPTER_MODEL_ID,
        token=hf_token,
    )
    model.eval()
    mem = ""
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated(0) / 1e9
        mem = f" | VRAM used: {used:.1f} GB"
    print(f"✅ Adapter loaded. Device: {next(model.parameters()).device}{mem}", flush=True)
    return model, tokenizer


def run_task_local(
    model, tokenizer, task_id: str
) -> Dict[str, Any]:
    """
    Run the trained local adapter against a single task.
    Mimics the same output format as run_task() for the hackathon validator.
    """
    import torch

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from server.environment import EmailTriageEnvironment

    env = EmailTriageEnvironment()
    obs = env.reset(seed=42, task_id=task_id)
    obs_dict = obs.model_dump()

    print(f"\n{'='*60}", flush=True)
    print(f"  Task: {task_id.upper()} [LOCAL ADAPTER: {ADAPTER_MODEL_ID}]", flush=True)
    print(f"  Emails: {obs_dict['inbox_stats'].get('total', 0)}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"[START] task={task_id} env={BENCHMARK} model={ADAPTER_MODEL_ID}", flush=True)

    step = 0
    max_steps = MAX_STEPS_PER_TASK.get(task_id, 40)
    step_rewards: List[float] = []

    while not obs_dict.get("done", False) and step < max_steps:
        if not obs_dict.get("emails"):
            break

        prompt = format_emails_for_prompt(obs_dict)

        # Build chat messages and tokenise
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(next(model.parameters()).device)

            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    max_new_tokens=MAX_TOKENS,
                    do_sample=False,
                    temperature=None,
                    pad_token_id=tokenizer.pad_token_id,
                )
            llm_text = tokenizer.decode(
                out[0][input_ids.shape[-1]:],
                skip_special_tokens=True,
            )
            action = parse_llm_response(llm_text)
        except Exception as e:
            print(f"  ⚠ Local generation error at step {step}: {e}", flush=True)
            action = None

        if action is None:
            action = make_fallback_action(obs_dict)

        action_str = json.dumps(action, separators=(",", ":")).replace("\n", " ")
        email_id = action.get("email_id", "?")
        category = action.get("category", "?")
        print(
            f"  Step {step+1}: email={email_id[:12]} → category={category}, "
            f"priority={action.get('priority', '?')}, dept={action.get('department', '?')}",
            flush=True,
        )

        obs = env.step(action)
        obs_dict = obs.model_dump()
        step += 1

        feedback = obs_dict.get("action_feedback", "")
        reward = obs_dict.get("step_reward", 0.0)
        done = obs_dict.get("done", False)
        step_rewards.append(reward)

        if feedback:
            print(f"         reward={reward:+.2f} | {feedback[:80]}", flush=True)
        print(
            f"[STEP] step={step} action={action_str} reward={reward:.2f} "
            f"done={str(done).lower()} error=null",
            flush=True,
        )

    grading = obs_dict.get("metadata", {}).get("grading", {})
    final_score = grading.get("final_score", 0.0)
    dimensions = grading.get("dimension_scores", {})
    steps_taken = grading.get("steps_taken", step)
    success = 0.0 < final_score < 1.0

    print(f"\n  ── Results for {task_id.upper()} ──", flush=True)
    print(f"  Final Score: {final_score:.4f}", flush=True)
    print(f"  Dimensions: {json.dumps(dimensions, indent=4)}", flush=True)
    print(f"  Emails processed: {grading.get('emails_processed', 0)}/{grading.get('emails_total', 0)}", flush=True)
    print(f"  Steps used: {steps_taken}/{grading.get('max_steps', max_steps)}", flush=True)

    clamped_score = min(max(final_score, 0.0001), 0.9999)
    rewards_str = ",".join(f"{r:.2f}" for r in step_rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps_taken} "
        f"score={clamped_score:.4f} rewards={rewards_str}",
        flush=True,
    )
    return grading


# ─────────────────────────────────────────────────────────────────────────
#  Main inference loop
# ─────────────────────────────────────────────────────────────────────────

def run_task(client: OpenAI, task_id: str) -> Dict[str, Any]:
    """Run the agent on a single task and return the grading result."""
    # Import environment directly (no HTTP needed)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from server.environment import EmailTriageEnvironment

    env = EmailTriageEnvironment()
    obs = env.reset(seed=42, task_id=task_id)
    obs_dict = obs.model_dump()

    print(f"\n{'='*60}", flush=True)
    print(f"  Task: {task_id.upper()} — {obs_dict.get('task_id', task_id)}", flush=True)
    print(f"  Emails: {obs_dict['inbox_stats'].get('total', 0)}", flush=True)
    print(f"  Max Steps: {MAX_STEPS_PER_TASK.get(task_id, 40)}", flush=True)
    print(f"{'='*60}", flush=True)

    # ── Structured output: task start ────────────────────────────
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    step = 0
    max_steps = MAX_STEPS_PER_TASK.get(task_id, 40)
    step_rewards: List[float] = []
    last_error: Optional[str] = None

    while not obs_dict.get("done", False) and step < max_steps:
        # Format observation for LLM
        prompt = format_emails_for_prompt(obs_dict)

        if not obs_dict.get("emails"):
            break  # No more emails to process

        last_error = None

        # Get LLM decision
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            llm_text = response.choices[0].message.content or ""
            action = parse_llm_response(llm_text)
        except Exception as e:
            last_error = str(e)
            print(f"  ⚠ LLM error at step {step}: {e}", flush=True)
            action = None

        if action is None:
            print(f"  ⚠ Could not parse LLM response at step {step}, using fallback", flush=True)
            action = make_fallback_action(obs_dict)

        # Build a compact one-line action string (no newlines)
        action_str = json.dumps(action, separators=(",", ":")).replace("\n", " ")

        # Step the environment
        email_id = action.get("email_id", "?")
        category = action.get("category", "?")
        print(f"  Step {step+1}: email={email_id[:12]} → category={category}, "
              f"priority={action.get('priority', '?')}, dept={action.get('department', '?')}", flush=True)

        obs = env.step(action)
        obs_dict = obs.model_dump()
        step += 1

        feedback = obs_dict.get("action_feedback", "")
        reward = obs_dict.get("step_reward", 0.0)
        done = obs_dict.get("done", False)
        step_rewards.append(reward)

        if feedback:
            print(f"         reward={reward:+.2f} | {feedback[:80]}", flush=True)

        # ── Structured output: per-step (exact required format) ───
        error_val = last_error if last_error else "null"
        done_val = str(done).lower()
        print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

    # Get final grading
    grading = obs_dict.get("metadata", {}).get("grading", {})
    final_score = grading.get("final_score", 0.0)
    dimensions = grading.get("dimension_scores", {})
    steps_taken = grading.get("steps_taken", step)
    success = 0.0 < final_score < 1.0

    print(f"\n  ── Results for {task_id.upper()} ──", flush=True)
    print(f"  Final Score: {final_score:.4f}", flush=True)
    print(f"  Dimensions: {json.dumps(dimensions, indent=4)}", flush=True)
    print(f"  Emails processed: {grading.get('emails_processed', 0)}/{grading.get('emails_total', 0)}", flush=True)
    print(f"  Steps used: {steps_taken}/{grading.get('max_steps', max_steps)}", flush=True)

    # ── Structured output: task end (exact required format) ───────
    # Validator requires score strictly in (0, 1) — clamp away from exact endpoints
    clamped_score = min(max(final_score, 0.0001), 0.9999)
    rewards_str = ",".join(f"{r:.2f}" for r in step_rewards)
    success_val = str(success).lower()
    print(f"[END] success={success_val} steps={steps_taken} score={clamped_score:.4f} rewards={rewards_str}", flush=True)

    return grading


def main():
    """Main entry point — run all 3 tasks and print summary."""
    print("=" * 60, flush=True)
    if USE_LOCAL_MODEL:
        print("  EMAIL TRIAGE — TRAINED ADAPTER INFERENCE", flush=True)
        print(f"  Adapter : {ADAPTER_MODEL_ID}", flush=True)
        print(f"  Base    : {BASE_MODEL_ID}", flush=True)
    else:
        print("  EMAIL TRIAGE ENVIRONMENT — BASELINE INFERENCE", flush=True)
        print(f"  Model   : {MODEL_NAME}", flush=True)
    print("=" * 60, flush=True)

    results = {}
    task_ids = ["easy", "medium", "hard"]

    # ── Local adapter mode ────────────────────────────────────────
    if USE_LOCAL_MODEL:
        print("\n🤗 Local adapter mode activated.", flush=True)
        print(f"   Loading {ADAPTER_MODEL_ID} on top of {BASE_MODEL_ID}...\n", flush=True)
        model, tokenizer = load_local_adapter()
        start_time = time.time()
        for task_id in task_ids:
            try:
                result = run_task_local(model, tokenizer, task_id)
                results[task_id] = result
            except Exception as e:
                print(f"\n✗ Task {task_id} failed: {e}", flush=True)
                print(f"[START] task={task_id} env={BENCHMARK} model={ADAPTER_MODEL_ID}", flush=True)
                print(f"[STEP] step=1 action=null reward=0.00 done=false error={e}", flush=True)
                print(f"[END] success=false steps=1 score=0.00 rewards=0.00", flush=True)
                results[task_id] = {"final_score": 0.0, "error": str(e)}
        elapsed = time.time() - start_time

        # Summary
        print(f"\n{'='*60}", flush=True)
        print("  SUMMARY (Trained Adapter)", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"  {'Task':<10} {'Score':>8} {'Emails':>10} {'Steps':>8}", flush=True)
        print(f"  {'─'*10} {'─'*8} {'─'*10} {'─'*8}", flush=True)
        for task_id in task_ids:
            r = results.get(task_id, {})
            score = r.get("final_score", 0.0)
            emails = f"{r.get('emails_processed', '?')}/{r.get('emails_total', '?')}"
            steps = f"{r.get('steps_taken', '?')}/{r.get('max_steps', '?')}"
            print(f"  {task_id:<10} {score:>8.4f} {emails:>10} {steps:>8}", flush=True)
        avg_score = sum(r.get("final_score", 0.0) for r in results.values()) / max(len(results), 1)
        print(f"\n  Average Score : {avg_score:.4f}", flush=True)
        print(f"  Total Time    : {elapsed:.1f}s", flush=True)
        print(f"{'='*60}\n", flush=True)
        return

    # ── API mode (default) ────────────────────────────────────────
    if not API_KEY:
        print("\n⚠ Warning: No API key found (HF_TOKEN or API_KEY)", flush=True)
        print("  Set HF_TOKEN or API_KEY environment variable.", flush=True)
        print("  Falling back to local-only mode (no LLM calls).\n", flush=True)

    if not MODEL_NAME:
        print("\n⚠ Warning: MODEL_NAME not set. Using default.", flush=True)

    # Initialise OpenAI client
    client = OpenAI(
        api_key=API_KEY or "dummy-key",
        base_url=API_BASE_URL,
    )

    results = {}
    task_ids = ["easy", "medium", "hard"]

    start_time = time.time()

    for task_id in task_ids:
        try:
            result = run_task(client, task_id)
            results[task_id] = result
        except Exception as e:
            print(f"\n✗ Task {task_id} failed: {e}", flush=True)
            # Emit structured blocks even on failure so the validator sees them
            print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)
            print(f"[STEP] step=1 action=null reward=0.00 done=false error={e}", flush=True)
            print(f"[END] success=false steps=1 score=0.00 rewards=0.00", flush=True)
            results[task_id] = {"final_score": 0.0, "error": str(e)}

    elapsed = time.time() - start_time

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("  SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  {'Task':<10} {'Score':>8} {'Emails':>10} {'Steps':>8}", flush=True)
    print(f"  {'─'*10} {'─'*8} {'─'*10} {'─'*8}", flush=True)

    for task_id in task_ids:
        r = results.get(task_id, {})
        score = r.get("final_score", 0.0)
        emails = f"{r.get('emails_processed', '?')}/{r.get('emails_total', '?')}"
        steps = f"{r.get('steps_taken', '?')}/{r.get('max_steps', '?')}"
        print(f"  {task_id:<10} {score:>8.4f} {emails:>10} {steps:>8}", flush=True)

    avg_score = sum(r.get("final_score", 0.0) for r in results.values()) / len(results)
    print(f"\n  Average Score: {avg_score:.4f}", flush=True)
    print(f"  Total Time: {elapsed:.1f}s", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()
