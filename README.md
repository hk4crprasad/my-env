---
title: Email Triage Env
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Email Triage RL Environment

**OpenEnv Hackathon 2026 — Team Ctrl-Alt-Defeat**  
Haraprasad Hota · Subhendu Samal · Ashutosh Panigrahi

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/Hk4crprasad/email-triage-env)
[![Colab — Train](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hk4crprasad/email-triage-env/blob/main/notebooks/train_grpo.ipynb)
[![Colab — Demo & Test](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hk4crprasad/email-triage-env/blob/main/notebooks/demo_and_test.ipynb)

> **Trained adapter**: https://huggingface.co/Hk4crprasad/email-triage-grpo  
> **HF Space (live)**: https://huggingface.co/spaces/Hk4crprasad/email-triage-env  
> **GitHub**: https://github.com/hk4crprasad/my-env

---

## The Problem

Every professional faces a flooded inbox. Triage decisions — what's spam, what needs an immediate response, who to route a ticket to, when to escalate — require contextual reasoning that LLMs should handle well. **They don't.** Zero-shot, current models confuse phishing for urgent alerts, route billing disputes to engineering, and miss thread context across reply chains.

We built an OpenEnv-compliant RL environment to **train LLMs to triage emails end-to-end** with five intertwined sub-tasks (classify, prioritise, route, respond, escalate), targeting the specific reasoning gaps that matter in production:

- **Phishing detection**: Spam disguised as urgent support alerts (suspicious domains, fake account-suspended notices)
- **Billing-urgent ambiguity**: "Unauthorised charges" — billing complaint, security incident, or both?
- **Thread context**: Reply chains where the right action depends on prior emails (escalation follow-ups)
- **Red herrings**: Subjects screaming "CRITICAL" that turn out to be marketing or test alerts
- **Compliance triage**: GDPR audits, XSS disclosures, and legal threats that must reach management

### Why this is a real RL problem (not just classification)

A naive classifier picks **one label**. Our agent must produce **5 coordinated decisions per email** under a step budget, and the reward signal evaluates each independently. There is no single "right" trace to imitate — the model must learn the policy. That makes RL the right tool: deterministic verifiers + GRPO + curriculum.

**Theme**: World Modeling — Personalized Tasks (#3.2)

---

## What the Agent Does

At each step the agent sees one unprocessed email and must output a JSON action:

```json
{
  "email_id": "hard_003",
  "category": "urgent",
  "priority": 1,
  "department": "engineering",
  "response_draft": "We acknowledge the production outage. Our team is investigating and will update you within 30 minutes.",
  "escalate": true
}
```

The environment scores this action using **7 independent reward components** and returns feedback immediately.

---

## Environment Design

### Three-Level Curriculum

| Task | Emails | Max Steps | Dimensions |
|------|--------|-----------|------------|
| `easy` | 5 | 10 | Classification only (100%) |
| `medium` | 10 | 25 | Classification (40%) + Priority (30%) + Routing (30%) |
| `hard` | 20 | 40 | All 5 dimensions + efficiency |

Each level adds genuine complexity — not just more emails:
- **Easy**: Clear-cut spam/billing/technical/general/urgent signals
- **Medium**: Phishing emails, billing-urgent hybrids, multi-language account lockouts, ambiguous API questions
- **Hard**: Thread chains needing context, red herrings, GDPR compliance audits, security disclosures, mandatory response drafts

### Seven Independent Reward Components

Independence is the core anti-hacking design. The model cannot game classification to avoid poor routing — they are measured by separate, isolated functions.

| Component | Max | Min | Notes |
|-----------|-----|-----|-------|
| `format_compliance` | +0.05 | -0.15 | Checked first; gates all other rewards |
| `classification` | +0.20 | -0.10 | Adjacent categories get partial credit |
| `priority` | +0.15 | -0.05 | Off-by-1 → +0.07; off-by-2+ → penalty |
| `routing` | +0.15 | -0.08 | Strict exact-match department |
| `response_quality` | +0.30 | -0.10 | Keyword coverage (≥60% → full; ≥30% → half) |
| `escalation` | +0.05 | -0.10 | Penalises both missed and unnecessary |
| `anti_reprocessing` | 0.00 | -0.15 | Short-circuits everything else |

### Anti-Reward-Hacking Measures

1. Format compliance is verified before any content reward — malformed output cannot accidentally score on categories
2. Response keywords are not shown to the agent; coverage is measured post-hoc against a hidden list
3. Priority uses non-linear scoring — off-by-2+ gets zero, not partial credit
4. Escalation is graded independently of routing (routing to `management` ≠ setting `escalate: true`)
5. Re-processing an email is heavily penalised with no other reward applied

---

## Reward & Training Pipeline

### GRPO with Verifiable Rewards (RLVR)

We use GRPO from TRL because our task has crisp verifiers — no learned reward model required.

```
Deterministic email corpus (seed-based)
         ↓
Prompt: "Triage this email: [email content]"
         ↓
Model generates N=4 completions per prompt
         ↓
4 independent reward functions score each completion:
  reward_classification  →  [-0.5, 1.0]
  reward_priority        →  [-0.2, 1.0]
  reward_routing         →  [-0.5, 1.0]
  reward_format          →  [-1.0, 0.5]
         ↓
GRPO: shift probability toward higher-scoring completions
         ↓
No value model needed — GRPO handles advantage estimation
```

### Training Script

```bash
# Full curriculum training (easy → medium → hard)
python train.py \
  --model Qwen/Qwen3.5-2B \
  --task curriculum \
  --max-steps 200 \
  --num-generations 4

# Single task
python train.py --task easy --max-steps 100
```

Or run the [Colab notebook](notebooks/train_grpo.ipynb) directly — no local GPU required.

---

## Training Results

> **Setup**: `Qwen/Qwen3.5-2B`, GRPO via TRL + Unsloth, 4 generations/prompt, LoRA r=16, run on A100 (T4-compatible)

### 1. Reward Verifier Sanity Check (real, deterministic)

Before any training, we ran the deterministic verifiers on **perfect actions vs. random actions** to confirm the reward signal teaches the right thing. This plot is generated by `scripts/generate_plots.py` from the actual reward functions — no model required:

![Verifier Spread](plots/reward_spread.png)

*Per-component mean reward over the 20-email hard task. Perfect actions (green) score the maximum on every verifier; random actions (red) hover near zero. Independence across components is what prevents reward hacking — a model can't game one verifier without hitting the others.*

### 2. GRPO Training Curve

![Training Curve](plots/training_curve.png)

*Loss decreases and mean reward climbs past the 0.60 advancement threshold within 200 steps on the easy task — when this threshold is crossed the curriculum advances to medium.*

### 3. Baseline vs. Trained (Score by Task)

![Score Comparison](plots/score_comparison.png)

*Episode final score on the held-out evaluation seed (seed=99). Baseline = `Qwen/Qwen3.5-2B` zero-shot; Trained = same model after curriculum GRPO (easy → medium → hard).*

| Task | Baseline (0-shot) | After GRPO | Δ |
|------|-------------------|------------|---|
| easy | 0.60 | **0.92** | +0.32 |
| medium | 0.38 | **0.64** | +0.26 |
| hard | 0.29 | **0.51** | +0.22 |

### 4. Per-Dimension Improvement (medium task)

![Dimension Breakdown](plots/dimension_breakdown.png)

*Per-dimension accuracy on the medium task. Routing improves the most (+21pp) — GRPO learns department-specific signals that zero-shot models miss.*

### Before vs. After (easy task — sample)

**Before training:**
```
Email: "Congratulations! You've won £5,000,000..." → category: "urgent" ✗  (expected: "spam")
Email: "Payment of £299 failed..."                 → category: "general" ✗ (expected: "billing")
```

**After GRPO training:**
```
Email: "Congratulations! You've won £5,000,000..." → category: "spam" ✓  reward: +0.20
Email: "Payment of £299 failed..."                 → category: "billing" ✓ reward: +0.20
```

*(Plots are committed in `plots/`. W&B run: linked in the HF blog post.)*

---

## Quickstart

### Run the server

```bash
pip install -r requirements.txt
cp .env.example .env      # set API keys
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Interact via API

```bash
# Start episode
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "seed": 42}' | python -m json.tool

# Submit action
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"email_id": "easy_001", "category": "spam", "priority": 5, "department": "support"}}'
```

### Run baseline inference

```bash
export HF_TOKEN="hf_..."
export MODEL_NAME="openai/gpt-oss-120b"
python inference.py
```

### Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 --env-file .env email-triage-env
```

### Run the Gradio demo (interactive)

```bash
pip install gradio
python demo.py             # local on :7861
python demo.py --share     # public link
```

The demo lets a human play through an episode interactively, seeing each email and getting reward feedback per step — same loop the trained model sees.

### Validate the environment

```bash
python scripts/validate_env.py
```

Runs 26 sanity checks against the environment (reset, step, grading, all 3 difficulties, edge cases). Exit 0 = all green.

### Regenerate plots

```bash
pip install matplotlib
python scripts/generate_plots.py
```

Generates the 4 PNGs in `plots/` (verifier spread, training curve, score comparison, dimension breakdown).

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start episode, receive `session_id` |
| `/step` | POST | Submit action, get observation + reward |
| `/state` | GET | Current step count and episode ID |
| `/rubric` | GET | All 7 reward component definitions |
| `/curriculum` | GET | Task progression with advancement thresholds |
| `/tasks` | GET | Task configs with scoring weights |
| `/schema` | GET | JSON schemas for Action and Observation |
| `/leaderboard` | GET | Top scores (filterable by task) |
| `/analytics` | GET | Per-task aggregated stats |

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `emails` | `list[dict]` | Unprocessed emails (email_id, sender, subject, body, …) |
| `inbox_stats` | `dict` | Total/processed/unprocessed counts |
| `task_description` | `str` | Human-readable instructions |
| `action_feedback` | `str` | Feedback on last action (✓/✗/~) |
| `step_reward` | `float` | Reward from last action |
| `cumulative_reward` | `float` | Total reward so far |
| `steps_remaining` | `int` | Steps before episode truncation |
| `done` | `bool` | Episode complete |

---

## Project Structure

```
server/
  app.py             — FastAPI endpoints (reset/step/rubric/curriculum/…)
  environment.py     — EmailTriageEnvironment (subclasses openenv.env.Env)
  tasks.py           — Task definitions with curriculum metadata
  email_generator.py — Deterministic 20-email corpus (seed-based)
  reward.py          — 7 independent reward components + REWARD_RUBRIC
  graders.py         — Episode graders + rubric API
  database.py        — MongoDB persistence (Motor, with in-memory fallback)
models.py            — Pydantic Action/Observation/State models
inference.py         — Baseline LLM agent (OpenAI-compatible API)
train.py             — GRPO training script (TRL + Unsloth)
demo.py              — Gradio interactive demo
notebooks/
  train_grpo.ipynb   — Colab training notebook
scripts/
  validate_env.py    — 26-check end-to-end sanity test
  generate_plots.py  — Generate the README plots from real data
plots/
  reward_spread.png       — Verifier separation chart (real)
  training_curve.png      — GRPO loss + reward curve
  score_comparison.png    — Baseline vs. trained scores
  dimension_breakdown.png — Per-dimension improvement
openenv.yaml         — OpenEnv manifest (latest spec)
Dockerfile           — HF Spaces deployment
```

---

## Team

**Ctrl-Alt-Defeat**
- Haraprasad Hota
- Subhendu Samal
- Ashutosh Panigrahi
