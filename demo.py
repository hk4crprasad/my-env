"""
Gradio demo for the Email Triage RL Environment.

Lets a human (or LLM) play through an episode interactively, seeing each
email, submitting an action, and watching reward feedback in real-time.
This is the storytelling interface for the hackathon demo.

Usage:
    python demo.py             # local
    python demo.py --share      # public link
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Python 3.14 compat: click.Choice lacks __class_getitem__; typer 0.24.x tries to
# use it as a generic base class. Adding the classmethod makes it subscriptable
# without affecting runtime behaviour (the subscript is only used for type hints).
try:
    import click as _click
    if not hasattr(_click.Choice, "__class_getitem__"):
        _click.Choice.__class_getitem__ = classmethod(lambda cls, x: cls)
except Exception:
    pass

import gradio as gr

from server.environment import EmailTriageEnvironment
from server.tasks import TASKS, list_task_ids
from server.reward import REWARD_RUBRIC


# ─────────────────────────────────────────────────────────────────────────
#  Session state (per-user via gr.State)
# ─────────────────────────────────────────────────────────────────────────

def new_env() -> EmailTriageEnvironment:
    return EmailTriageEnvironment()


def reset_episode(env: EmailTriageEnvironment, task_id: str, seed: int) -> Tuple[Dict, str, str, float, str]:
    """Start a new episode and return the first email + status."""
    obs = env.reset(task_id=task_id, seed=int(seed))
    obs_dict = obs.model_dump()

    if not obs_dict.get("emails"):
        return obs_dict, "(empty inbox)", "", 0.0, "Episode started."

    email = obs_dict["emails"][0]
    email_view = format_email(email)
    status = f"Task: **{task_id.upper()}** · {len(obs_dict['emails'])} emails to triage · {obs_dict['steps_remaining']} steps"
    return obs_dict, email_view, email["email_id"], 0.0, status


def format_email(email: Dict[str, Any]) -> str:
    """Render an email as Markdown."""
    thread = f"\n**Thread**: `{email['thread_id']}`" if email.get("thread_id") else ""
    return (
        f"### 📧 {email.get('subject', '(no subject)')}\n\n"
        f"**From**: {email.get('sender_name', '')} `<{email.get('sender', '')}>`  \n"
        f"**Email ID**: `{email['email_id']}`  \n"
        f"**Time**: {email.get('timestamp', '')}  \n"
        f"**Reply**: {'Yes' if email.get('is_reply') else 'No'} · "
        f"**Attachment**: {'Yes' if email.get('has_attachment') else 'No'}{thread}\n\n"
        f"---\n\n"
        f"{email.get('body', '(empty body)')}"
    )


def submit_action(
    env: EmailTriageEnvironment,
    obs_state: Dict,
    email_id: str,
    category: str,
    priority: int,
    department: str,
    response_draft: str,
    escalate: bool,
) -> Tuple[Dict, str, str, float, str, str]:
    """Submit an action and return updated observation + email + feedback."""
    if obs_state is None or obs_state.get("done"):
        return obs_state, "(no active episode)", "", 0.0, "Click **Reset** to start.", ""

    action = {
        "email_id": email_id,
        "category": category,
        "priority": int(priority),
        "department": department,
        "response_draft": response_draft.strip() or None,
        "escalate": bool(escalate),
    }

    obs = env.step(action)
    obs_dict = obs.model_dump()

    feedback = obs_dict.get("action_feedback", "")
    step_reward = obs_dict.get("step_reward", 0.0)
    cumulative = obs_dict.get("cumulative_reward", 0.0)
    done = obs_dict.get("done", False)

    if done:
        grading = obs_dict.get("metadata", {}).get("grading", {})
        final = grading.get("final_score", 0.0)
        dims = grading.get("dimension_scores", {})
        dims_str = ", ".join(f"{k}={v:.2f}" for k, v in dims.items())
        status = (
            f"🏁 **Episode complete!** Final score: **{final:.4f}**  \n"
            f"Per-dimension: {dims_str}  \n"
            f"Cumulative reward: {cumulative:.2f}"
        )
        next_email = "**(episode finished — click Reset to start a new one)**"
        next_id = ""
    else:
        next_email_data = obs_dict["emails"][0] if obs_dict.get("emails") else None
        next_email = format_email(next_email_data) if next_email_data else "(no more emails)"
        next_id = next_email_data["email_id"] if next_email_data else ""
        status = (
            f"Step {obs_dict.get('metadata', {}).get('step_count', '?')} · "
            f"Cumulative: **{cumulative:+.2f}** · "
            f"Steps remaining: {obs_dict.get('steps_remaining', '?')}"
        )

    feedback_md = f"**Step reward**: `{step_reward:+.2f}`  \n**Feedback**: {feedback}"
    return obs_dict, next_email, next_id, cumulative, status, feedback_md


# ─────────────────────────────────────────────────────────────────────────
#  Build UI
# ─────────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    rubric_md = "## Reward Rubric (7 independent components)\n\n"
    for k, v in REWARD_RUBRIC.items():
        rubric_md += f"- **{k}**: {v['description']} (range: `{v['min_reward']:+.2f}` … `{v['max_reward']:+.2f}`)\n"

    with gr.Blocks(theme=gr.themes.Soft(), title="Email Triage RL") as demo:
        gr.Markdown(
            "# 📧 Email Triage RL Environment\n"
            "**OpenEnv Hackathon 2026 — Team Ctrl-Alt-Defeat**  \n"
            "Live demo: classify, prioritise, route, and respond to emails — get reward feedback per step.  \n"
            "[Source on HF Spaces](https://huggingface.co/spaces/Hk4crprasad/email-triage-env)"
        )

        env_state = gr.State(value=None)
        obs_state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Setup")
                task_id = gr.Dropdown(list_task_ids(), value="easy", label="Task")
                seed = gr.Number(value=42, label="Seed", precision=0)
                reset_btn = gr.Button("🔄 Reset Episode", variant="primary")
                gr.Markdown(rubric_md)

            with gr.Column(scale=2):
                gr.Markdown("### Inbox")
                status_md = gr.Markdown("Click **Reset Episode** to start.")
                email_md = gr.Markdown("(no email loaded)", elem_id="email-card")

                gr.Markdown("### Your Action")
                with gr.Row():
                    email_id = gr.Textbox(label="Email ID", interactive=False)
                with gr.Row():
                    category = gr.Dropdown(
                        ["spam", "billing", "technical", "general", "urgent"],
                        value="general", label="Category"
                    )
                    priority = gr.Slider(1, 5, value=3, step=1, label="Priority (1=critical)")
                    department = gr.Dropdown(
                        ["engineering", "billing", "support", "management"],
                        value="support", label="Department"
                    )
                with gr.Row():
                    escalate = gr.Checkbox(label="Escalate to management")
                response_draft = gr.Textbox(
                    label="Response draft (only required for urgent emails on hard task)",
                    lines=3, placeholder="(optional)"
                )
                submit_btn = gr.Button("📨 Submit Action", variant="primary")
                feedback_md = gr.Markdown("(no action yet)")
                cumulative = gr.Number(label="Cumulative reward", value=0.0, interactive=False)

        # Wire up callbacks
        def on_init():
            env = new_env()
            return env

        demo.load(fn=on_init, inputs=None, outputs=env_state)

        reset_btn.click(
            fn=reset_episode,
            inputs=[env_state, task_id, seed],
            outputs=[obs_state, email_md, email_id, cumulative, status_md],
        )

        submit_btn.click(
            fn=submit_action,
            inputs=[env_state, obs_state, email_id, category, priority,
                     department, response_draft, escalate],
            outputs=[obs_state, email_md, email_id, cumulative, status_md, feedback_md],
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Public Gradio link")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
