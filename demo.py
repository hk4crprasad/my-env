"""
Gradio demo for the Email Triage RL Environment.

Two tabs:
  1. "Triage one email"   — manual play: human picks the action, sees per-step reward
  2. "Baseline vs Trained"— side-by-side inference: same email → base Qwen2.5-3B vs
                            our GRPO-trained LoRA adapter (Hk4crprasad/email-triage-grpo)

Designed to be embedded inside the FastAPI server (mounted at /demo on the same
port as the OpenEnv API) AND to run standalone via `python demo.py`.

Usage:
    python demo.py             # standalone on :7861
    python demo.py --share     # public Gradio link

Or, when imported:
    from demo import build_ui
    blocks = build_ui()
    app = gr.mount_gradio_app(fastapi_app, blocks, path="/demo")
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from typing import Any, Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Python 3.14 compat: click.Choice lacks __class_getitem__; typer 0.24.x tries to
# use it as a generic base class. Patch makes it subscriptable for typing only.
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


# Trained adapter on HF Hub (LoRA over Qwen2.5-3B-Instruct, ~43 MB)
BASE_MODEL_ID    = os.environ.get("BASE_MODEL_ID",    "Qwen/Qwen2.5-3B-Instruct")
ADAPTER_MODEL_ID = os.environ.get("ADAPTER_MODEL_ID", "Hk4crprasad/email-triage-grpo")
HF_TOKEN         = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY")


# ─────────────────────────────────────────────────────────────────────────
#  Manual-play tab helpers
# ─────────────────────────────────────────────────────────────────────────

def new_env() -> EmailTriageEnvironment:
    return EmailTriageEnvironment()


def reset_episode(env: EmailTriageEnvironment, task_id: str, seed: int) -> Tuple[Dict, str, str, float, str]:
    obs = env.reset(task_id=task_id, seed=int(seed))
    od  = obs.model_dump()
    if not od.get("emails"):
        return od, "(empty inbox)", "", 0.0, "Episode started."
    email = od["emails"][0]
    status = (
        f"Task: **{task_id.upper()}** · "
        f"{len(od['emails'])} emails · "
        f"{od['steps_remaining']} steps remaining"
    )
    return od, format_email(email), email["email_id"], 0.0, status


def format_email(email: Dict[str, Any]) -> str:
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
    if obs_state is None or obs_state.get("done"):
        return obs_state, "(no active episode)", "", 0.0, "Click **Reset** to start.", ""

    action = {
        "email_id":       email_id,
        "category":       category,
        "priority":       int(priority),
        "department":     department,
        "response_draft": response_draft.strip() or None,
        "escalate":       bool(escalate),
    }
    obs = env.step(action)
    od  = obs.model_dump()

    feedback     = od.get("action_feedback", "")
    step_reward  = od.get("step_reward", 0.0)
    cumulative   = od.get("cumulative_reward", 0.0)
    done         = od.get("done", False)

    if done:
        g = od.get("metadata", {}).get("grading", {})
        final = g.get("final_score", 0.0)
        dims  = g.get("dimension_scores", {})
        dims_str = ", ".join(f"{k}={v:.2f}" for k, v in dims.items())
        status = (
            f"🏁 **Episode complete!** Final score: **{final:.4f}**  \n"
            f"Per-dimension: {dims_str}  \n"
            f"Cumulative reward: {cumulative:.2f}"
        )
        next_email = "**(episode finished — click Reset to start a new one)**"
        next_id    = ""
    else:
        nxt = od["emails"][0] if od.get("emails") else None
        next_email = format_email(nxt) if nxt else "(no more emails)"
        next_id    = nxt["email_id"] if nxt else ""
        status = (
            f"Step {od.get('metadata', {}).get('step_count', '?')} · "
            f"Cumulative: **{cumulative:+.2f}** · "
            f"Steps remaining: {od.get('steps_remaining', '?')}"
        )

    feedback_md = f"**Step reward**: `{step_reward:+.2f}`  \n**Feedback**: {feedback}"
    return od, next_email, next_id, cumulative, status, feedback_md


# ─────────────────────────────────────────────────────────────────────────
#  Adapter-vs-baseline tab — lazy load, toggle adapter, run inference
# ─────────────────────────────────────────────────────────────────────────

_MODEL_LOCK   = threading.Lock()
_MODEL_STATE: Dict[str, Any] = {"model": None, "tokenizer": None, "loaded": False}


def _load_adapter_lazy() -> Tuple[Any, Any, str]:
    """Load Qwen2.5-3B + LoRA adapter once, cached. Returns (model, tokenizer, info)."""
    if _MODEL_STATE["loaded"]:
        return _MODEL_STATE["model"], _MODEL_STATE["tokenizer"], _MODEL_STATE["info"]
    with _MODEL_LOCK:
        if _MODEL_STATE["loaded"]:
            return _MODEL_STATE["model"], _MODEL_STATE["tokenizer"], _MODEL_STATE["info"]
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel
        except ImportError as e:
            raise gr.Error(
                f"Adapter inference requires transformers + peft + bitsandbytes: {e}"
            )

        if not torch.cuda.is_available():
            raise gr.Error(
                "No GPU detected. Adapter inference needs CUDA "
                "(this Space must be on the GPU tier)."
            )

        props   = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
        cc_major, cc_minor = props.major, props.minor
        print(f"🔍 GPU detected: {props.name} (cc={cc_major}.{cc_minor}, VRAM={vram_gb:.1f} GB)", flush=True)
        print(f"    CUDA version: {torch.version.cuda}  |  torch: {torch.__version__}", flush=True)

        info_lines = [f"Loading {BASE_MODEL_ID} …"]
        print(f"📥 Downloading tokenizer: {BASE_MODEL_ID} ...",flush=True)
        tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        print(f"✅ Tokenizer loaded",flush=True)

        # ── Dispatch by compute capability AND VRAM ─────────────────────
        supports_bf16 = cc_major >= 8
        use_4bit = (not supports_bf16) or (vram_gb < 12)

        if use_4bit:
            print(f"📥 Downloading base model in 4-bit NF4: {BASE_MODEL_ID} ...",flush=True)
            bnb = BitsAndBytesConfig(
                load_in_4bit             = True,
                bnb_4bit_quant_type      = "nf4",
                bnb_4bit_compute_dtype   = torch.float16,
                bnb_4bit_use_double_quant= True,
            )
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID, quantization_config=bnb, device_map="auto",
                token=HF_TOKEN, trust_remote_code=True,
            )
            info_lines.append(
                f"GPU: {props.name} cc={cc_major}.{cc_minor} VRAM={vram_gb:.1f} GB → 4-bit NF4"
            )
        else:
            print(f"📥 Downloading base model in bfloat16: {BASE_MODEL_ID} ...",flush=True)
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
                token=HF_TOKEN, trust_remote_code=True,
            )
            info_lines.append(
                f"GPU: {props.name} cc={cc_major}.{cc_minor} VRAM={vram_gb:.1f} GB → bfloat16"
            )
        print(f"✅ Base model loaded",flush=True)

        print(f"📥 Downloading LoRA adapter: {ADAPTER_MODEL_ID} ...",flush=True)
        info_lines.append(f"Loading LoRA adapter {ADAPTER_MODEL_ID} …")
        model = PeftModel.from_pretrained(base, ADAPTER_MODEL_ID, token=HF_TOKEN)
        model.eval()
        print(f"✅ Adapter merged · device={next(model.parameters()).device}",flush=True)
        info_lines.append(f"✅ Loaded · device={next(model.parameters()).device}")

        info = " | ".join(info_lines)
        _MODEL_STATE.update({"model": model, "tokenizer": tok, "info": info, "loaded": True})
        return model, tok, info


def _build_inference_prompt(email: Dict[str, Any], task_description: str) -> Tuple[list, str]:
    """Mirror inference.py / train.py prompting for parity."""
    from train import SYSTEM_PROMPT  # must match what the adapter was trained with
    from train import format_email_prompt
    user = format_email_prompt(email, task_description)
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user},
    ]
    return msgs, user


def _generate(model, tokenizer, msgs: list, use_adapter: bool) -> str:
    import torch
    if use_adapter:
        # ensure adapter is active
        if hasattr(model, "enable_adapter_layers"):
            model.enable_adapter_layers()
    else:
        # disable adapter → pure base-model behaviour
        if hasattr(model, "disable_adapter_layers"):
            model.disable_adapter_layers()

    inputs = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens   = 320,
            do_sample        = False,
            temperature      = None,
            pad_token_id     = tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)
    if use_adapter and hasattr(model, "enable_adapter_layers"):
        model.enable_adapter_layers()  # leave model in adapter-on state by default
    return text.strip()


def _score_action(action: Dict[str, Any], task_id: str, seed: int) -> Tuple[float, str, Dict[str, Any]]:
    """Run a one-step env episode with `action` to get reward + ground truth."""
    from server.email_generator import generate_emails
    from server.reward import compute_step_reward
    emails, gts = generate_emails(task_id, seed)
    truth = next((g for g in gts if g.email_id == action.get("email_id", "")), None)
    if truth is None:
        return 0.0, "✗ unknown email_id", {}
    valid_ids = {e.email_id for e in emails}
    task = TASKS[task_id]
    reward, fb = compute_step_reward(
        action, truth,
        requires_priority = task.requires_priority,
        requires_routing  = task.requires_routing,
        requires_response = task.requires_response,
        already_processed = False,
        valid_email_ids   = valid_ids,
    )
    return reward, fb, {
        "category":   truth.category,
        "priority":   truth.priority,
        "department": truth.department,
        "should_escalate": truth.priority == 1 or truth.department == "management",
    }


def run_compare(task_id: str, seed: int, email_index: int):
    """Generator: yields live status updates to the UI as the model loads and runs."""
    print(f"🚀 run_compare START: task={task_id} seed={seed} idx={email_index}", flush=True)

    # ── Step 1: load email ────────────────────────────────────────────────
    yield "", "", "", "", "⏳ **Step 1/4** — Loading email from environment…"
    try:
        seed = int(seed); email_index = int(email_index)
    except (ValueError, TypeError):
        yield "Invalid seed/index", "", "", "", "❌ seed/index must be integers"
        return

    try:
        env = EmailTriageEnvironment()
        obs = env.reset(task_id=task_id, seed=seed)
        od  = obs.model_dump()
    except Exception as e:
        print(f"❌ env.reset() failed: {type(e).__name__}: {e}", flush=True)
        yield f"❌ env.reset() error: {e}", "", "", "", "❌ Environment error"
        return

    emails = od.get("emails", [])
    if email_index < 0 or email_index >= len(emails):
        yield f"Index {email_index} out of range (0..{len(emails)-1})", "", "", "", "❌ Index out of range"
        return

    email    = emails[email_index]
    email_md = format_email(email)
    print(f"📨 Email loaded: {email.get('email_id')}", flush=True)

    # ── Step 2: GPU check + model load ───────────────────────────────────
    try:
        import torch
        if not torch.cuda.is_available():
            msg = "❌ No GPU — set `hardware: t4-small` in README.md and restart Space"
            print(msg, flush=True)
            yield email_md, "", "", "", msg
            return
        props   = torch.cuda.get_device_properties(0)
        gpu_str = f"{props.name}  cc={props.major}.{props.minor}  VRAM={props.total_memory/1e9:.1f} GB"
    except Exception as e:
        yield email_md, "", "", "", f"❌ torch error: {e}"
        return

    if _MODEL_STATE["loaded"]:
        status = f"⚡ **Step 2/4** — Model already in VRAM ({gpu_str}) — running instantly"
    else:
        status = (
            f"⏳ **Step 2/4** — Downloading model to GPU ({gpu_str})\n\n"
            f"First run downloads **~3 GB** — expect **60–90 s** on T4.\n"
            f"Watch the container **Logs** tab for progress. Subsequent runs are instant."
        )
    print(f"🖥  {gpu_str}", flush=True)
    yield email_md, "", "", "", status

    try:
        model, tokenizer, info = _load_adapter_lazy()
    except (gr.Error, Exception) as e:
        print(f"❌ Model loading failed: {type(e).__name__}: {e}", flush=True)
        yield email_md, "", "", "", f"❌ Model loading failed: {e}"
        return

    # ── Step 3: baseline inference ────────────────────────────────────────
    yield email_md, "*(running…)*", "*(waiting…)*", "", f"⏳ **Step 3/4** — Running baseline (adapter OFF)…"
    print("🔵 Generating baseline output…", flush=True)
    msgs, _ = _build_inference_prompt(email, od.get("task_description", ""))
    base_text = _generate(model, tokenizer, msgs, use_adapter=False)

    # ── Step 4: trained inference ─────────────────────────────────────────
    yield email_md, "*(done)*", "*(running…)*", "", f"⏳ **Step 4/4** — Running trained model (adapter ON)…"
    print("🟢 Generating trained output…", flush=True)
    train_text = _generate(model, tokenizer, msgs, use_adapter=True)

    def _to_action(text: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(text)
        except Exception:
            for i, ch in enumerate(text):
                if ch == "{":
                    depth = 0
                    for j in range(i, len(text)):
                        if text[j] == "{": depth += 1
                        elif text[j] == "}":
                            depth -= 1
                            if depth == 0:
                                try: return json.loads(text[i:j+1])
                                except Exception: return None
            return None

    base_action  = _to_action(base_text)  or {"email_id": email["email_id"], "category": "general"}
    train_action = _to_action(train_text) or {"email_id": email["email_id"], "category": "general"}

    base_reward,  base_fb,  truth = _score_action(base_action,  task_id, seed)
    train_reward, train_fb, _     = _score_action(train_action, task_id, seed)

    base_md = (
        f"### 🔵 Baseline (Qwen2.5-3B, 0-shot)\n\n"
        f"```json\n{json.dumps(base_action, indent=2)}\n```\n\n"
        f"**Reward**: `{base_reward:+.2f}`  \n"
        f"**Feedback**: {base_fb}"
    )
    train_md = (
        f"### 🟢 Trained (after GRPO)\n\n"
        f"```json\n{json.dumps(train_action, indent=2)}\n```\n\n"
        f"**Reward**: `{train_reward:+.2f}`  \n"
        f"**Feedback**: {train_fb}"
    )
    truth_md = (
        f"### 🎯 Ground truth\n\n"
        f"- **category**: `{truth.get('category', '?')}`\n"
        f"- **priority**: `{truth.get('priority', '?')}`\n"
        f"- **department**: `{truth.get('department', '?')}`\n"
        f"- **should_escalate**: `{truth.get('should_escalate', '?')}`\n\n"
        f"Δ reward (trained − baseline): **{train_reward - base_reward:+.2f}**"
    )

    yield email_md, base_md, train_md, truth_md, info


# ─────────────────────────────────────────────────────────────────────────
#  Build UI
# ─────────────────────────────────────────────────────────────────────────

def _gpu_status_md() -> str:
    """Compute GPU status string once at UI build time."""
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return (
                f"> 🖥 **GPU ready:** {p.name}  |  "
                f"cc={p.major}.{p.minor}  |  "
                f"VRAM={p.total_memory/1e9:.1f} GB  |  "
                f"CUDA {torch.version.cuda}  ✅  \n"
                f"> Model is **not loaded yet** — click ▶ Run to start download (~60 s first run)."
            )
        return "> ⚠ **No GPU detected.** Adapter inference requires T4. Check Space hardware tier."
    except Exception as e:
        return f"> ⚠ Could not detect GPU: `{e}`"


def build_ui() -> gr.Blocks:
    rubric_md = "## Reward Rubric (8 independent components)\n\n"
    for k, v in REWARD_RUBRIC.items():
        rubric_md += (
            f"- **{k}**: {v['description']} "
            f"(range: `{v['min_reward']:+.2f}` … `{v['max_reward']:+.2f}`)\n"
        )
    gpu_md = _gpu_status_md()

    with gr.Blocks(title="Email Triage RL") as demo:
        gr.Markdown(
            "# 📧 Email Triage RL Environment\n"
            "**OpenEnv Hackathon 2026 — Team Ctrl-Alt-Defeat**  \n"
            "Train an LLM to triage emails. Live API: `/reset`, `/step`, `/rubric` — "
            "this Gradio UI runs on the same port.  \n"
            "[HF Space](https://huggingface.co/spaces/Hk4crprasad/email-triage-env) · "
            "[Trained adapter](https://huggingface.co/Hk4crprasad/email-triage-grpo) · "
            "[GitHub](https://github.com/hk4crprasad/my-env)"
        )

        with gr.Tabs():
            # ──────────────────────────────────────────────────────────────
            #  Tab 1 — manual play
            # ──────────────────────────────────────────────────────────────
            with gr.Tab("🎮 Triage one email"):
                env_state = gr.State(value=None)
                obs_state = gr.State(value=None)

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Setup")
                        task_id = gr.Dropdown(list_task_ids(), value="easy", label="Task")
                        seed    = gr.Number(value=42, label="Seed", precision=0)
                        reset_btn = gr.Button("🔄 Reset Episode", variant="primary")
                        gr.Markdown(rubric_md)

                    with gr.Column(scale=2):
                        gr.Markdown("### Inbox")
                        status_md = gr.Markdown("Click **Reset Episode** to start.")
                        email_md  = gr.Markdown("(no email loaded)")

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
                        cumulative  = gr.Number(label="Cumulative reward", value=0.0, interactive=False)

                demo.load(fn=lambda: new_env(), inputs=None, outputs=env_state)

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

            # ──────────────────────────────────────────────────────────────
            #  Tab 2 — adapter vs baseline (live, on this Space's GPU)
            # ──────────────────────────────────────────────────────────────
            with gr.Tab("🆚 Baseline vs Trained adapter"):
                gr.Markdown(
                    "### Side-by-side inference\n"
                    "Loads `Qwen/Qwen2.5-3B-Instruct` once, then toggles the LoRA "
                    f"adapter `{ADAPTER_MODEL_ID}` on/off. Both runs see the **same** email; "
                    "their actions are scored by the live reward function and shown next to ground truth."
                )
                # GPU status shown immediately on page load — no click needed
                gr.Markdown(gpu_md)
                with gr.Row():
                    with gr.Column(scale=1):
                        cmp_task  = gr.Dropdown(list_task_ids(), value="hard", label="Task")
                        cmp_seed  = gr.Number(value=42, label="Seed", precision=0)
                        cmp_idx   = gr.Number(value=0, label="Email index (0-based)", precision=0)
                        run_btn   = gr.Button("▶ Run side-by-side", variant="primary")
                        info_md   = gr.Markdown("*(click ▶ Run to start)*")
                    with gr.Column(scale=2):
                        cmp_email = gr.Markdown("(no email loaded)")

                with gr.Row():
                    cmp_base  = gr.Markdown("*(baseline output will appear here)*")
                    cmp_train = gr.Markdown("*(trained output will appear here)*")
                cmp_truth = gr.Markdown()

                run_btn.click(
                    fn=run_compare,
                    inputs=[cmp_task, cmp_seed, cmp_idx],
                    outputs=[cmp_email, cmp_base, cmp_train, cmp_truth, info_md],
                    concurrency_limit=1,  # one GPU inference at a time (Gradio 5/6)
                )

    return demo


# ─────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Public Gradio link")
    parser.add_argument("--port",  type=int, default=7861)
    args = parser.parse_args()
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
