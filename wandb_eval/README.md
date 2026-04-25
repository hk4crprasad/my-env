# W&B Evaluation Suite — Email Triage RL Environment

**OpenEnv Hackathon 2026 · Team Ctrl-Alt-Defeat**

This folder contains a single evaluation script (`judge_eval.py`) that lets judges independently verify every claim made in the README and slides.

---

## What It Does

The script runs in five sections, all logged to a public W&B run:

| Section | What it tests |
|---------|--------------|
| 1. Environment Validation | All 3 tasks reset/step with perfect actions — confirms grading is deterministic |
| 2. Reward Component Analysis | Scores every hard-task email with perfect / random / adversarial actions — proves the 7 verifiers are independent and separating |
| 3. Baseline LLM | Runs any OpenAI-compatible model (free via HF Router) — no GPU needed |
| 4. Trained Adapter | Runs `Hk4crprasad/email-triage-grpo` for before/after comparison |
| 5. W&B Upload | Logs tables, charts, per-email breakdowns, and all 4 plots to W&B |

---

## Quick Start

### Option A — Environment validation only (no API key, no GPU, 2 min)

```bash
cd /path/to/my-env
pip install wandb
python wandb_eval/judge_eval.py --mode env
```

### Option B — Full baseline + environment (needs HF token, no GPU, ~5 min)

```bash
export HF_TOKEN="hf_..."
pip install wandb openai
python wandb_eval/judge_eval.py --mode full
```

### Option C — Full comparison with trained adapter (run in Colab — recommended)

Open the notebook directly:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hk4crprasad/email-triage-env/blob/main/notebooks/demo_and_test.ipynb)

The Colab notebook installs all dependencies (including CUDA torch), loads the trained adapter from HF Hub, runs all 3 tasks, and prints the before/after score table.

### Option D — Use your own LLM endpoint

```bash
python wandb_eval/judge_eval.py --mode baseline \
    --api-base https://api.openai.com/v1 \
    --api-key sk-... \
    --model gpt-4o-mini
```

---

## W&B Panels You'll See

After running, the W&B run contains:

- **`validation/results_table`** — pass/fail per task, final score, steps used
- **`reward_analysis/component_table`** — per-verifier mean rewards for perfect / random / adversarial actions
- **`reward_analysis/per_email_table`** — every hard-task email scored individually
- **`baseline/*`** — per-task scores + per-step table (category, priority, routing, feedback, latency)
- **`trained/*`** — same for the GRPO adapter
- **`improvement/summary_table`** — delta table: baseline → trained per task
- **`plots/*`** — all 4 committed PNG plots (reward spread, score comparison, dimension breakdown, training curve)

---

## Expected Output (env mode, no LLM)

```
────────────────────────────────────────────────────────────
  SECTION 1: Environment Validation
────────────────────────────────────────────────────────────
  [✓ PASS] easy     score=1.0000  steps=5   avg_reward=+0.550
  [✓ PASS] medium   score=0.9800  steps=10  avg_reward=+0.421
  [✓ PASS] hard     score=0.9600  steps=20  avg_reward=+0.388

  All 3 tasks PASS ✅

────────────────────────────────────────────────────────────
  SECTION 2: Reward Component Analysis
────────────────────────────────────────────────────────────
  Component        Perfect    Random   Adversarial     Max
  ──────────────── ───────── ───────── ────────────  ──────
  classification   +0.200    -0.012     -0.050        +0.20
  priority         +0.150    +0.030     +0.070        +0.15
  routing          +0.150    -0.005     -0.050        +0.15
  response         +0.211    +0.012     +0.000        +0.30
  escalation       +0.039    -0.008     -0.073        +0.05
  format           +0.050    +0.050     +0.050        +0.05
```

The separation between perfect and random/adversarial proves the verifiers are teaching meaningful signal.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | Hugging Face token (API + adapter download) |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `openai/gpt-oss-120b` | Model identifier |
| `WANDB_PROJECT` | `email-triage-eval` | W&B project |
| `WANDB_ENTITY` | your username | W&B entity/team |

---

## CLI Reference

```
python wandb_eval/judge_eval.py [OPTIONS]

Options:
  --mode {env,baseline,full}   Evaluation depth (default: full)
  --seed INT                   Random seed (default: 42)
  --api-base URL               LLM API base URL
  --model STR                  LLM model name
  --api-key STR                API key
  --use-local-model            Load trained LoRA adapter locally (GPU required)
  --adapter-id STR             HF Hub adapter ID (default: Hk4crprasad/email-triage-grpo)
  --base-model STR             Base model for adapter (default: Qwen/Qwen2.5-3B-Instruct)
  --wandb-project STR          W&B project name
  --wandb-entity STR           W&B entity/team
  --no-wandb                   Skip W&B upload (print to stdout only)
```

---

## Links

| Resource | URL |
|----------|-----|
| HF Space (live API) | https://huggingface.co/spaces/Hk4crprasad/email-triage-env |
| Trained adapter | https://huggingface.co/Hk4crprasad/email-triage-grpo |
| GitHub | https://github.com/hk4crprasad/my-env |
| Blog post | https://huggingface.co/blog/Hk4crprasad/email-triage-grpo-blog |
| Training Colab | notebooks/train_grpo.ipynb |
