"""
Inference Script — Email Triage Environment
=============================================

MANDATORY:
- Before submitting, ensure the following variables are defined in your
  environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root
  directory of the project.
- Participants must use OpenAI Client for all LLM calls using above variables.

This script runs a baseline LLM agent against all 3 tasks (easy, medium, hard)
and prints reproducible scores.
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
)

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
    print(f"[START] task={task_id}", flush=True)

    step = 0
    max_steps = MAX_STEPS_PER_TASK.get(task_id, 40)
    cumulative_reward = 0.0

    while not obs_dict.get("done", False) and step < max_steps:
        # Format observation for LLM
        prompt = format_emails_for_prompt(obs_dict)

        if not obs_dict.get("emails"):
            break  # No more emails to process

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
            print(f"  ⚠ LLM error at step {step}: {e}", flush=True)
            action = None

        if action is None:
            print(f"  ⚠ Could not parse LLM response at step {step}, using fallback", flush=True)
            action = make_fallback_action(obs_dict)

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
        cumulative_reward += reward
        if feedback:
            print(f"         reward={reward:+.2f} | {feedback[:80]}", flush=True)

        # ── Structured output: per-step ───────────────────────────
        print(f"[STEP] step={step} reward={reward:.4f}", flush=True)

    # Get final grading
    grading = obs_dict.get("metadata", {}).get("grading", {})
    final_score = grading.get("final_score", 0.0)
    dimensions = grading.get("dimension_scores", {})
    steps_taken = grading.get("steps_taken", step)

    print(f"\n  ── Results for {task_id.upper()} ──", flush=True)
    print(f"  Final Score: {final_score:.4f}", flush=True)
    print(f"  Dimensions: {json.dumps(dimensions, indent=4)}", flush=True)
    print(f"  Emails processed: {grading.get('emails_processed', 0)}/{grading.get('emails_total', 0)}", flush=True)
    print(f"  Steps used: {steps_taken}/{grading.get('max_steps', max_steps)}", flush=True)

    # ── Structured output: task end ───────────────────────────────
    print(f"[END] task={task_id} score={final_score:.4f} steps={steps_taken}", flush=True)

    return grading


def main():
    """Main entry point — run all 3 tasks and print summary."""
    print("=" * 60, flush=True)
    print("  EMAIL TRIAGE ENVIRONMENT — BASELINE INFERENCE", flush=True)
    print("=" * 60, flush=True)

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
            print(f"[START] task={task_id}", flush=True)
            print(f"[STEP] step=1 reward=0.0000", flush=True)
            print(f"[END] task={task_id} score=0.0000 steps=1", flush=True)
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
