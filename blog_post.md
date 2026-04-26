---
title: "We Trained an LLM to Triage Emails with GRPO — Email Triage RL Environment"
thumbnail: https://huggingface.co/spaces/Hk4crprasad/email-triage-env/resolve/main/plots/score_comparison.png
authors:
  - user: Hk4crprasad
  - user: subhendusamal
  - user: ashutoshpanigrahi
tags:
  - rl
  - grpo
  - openenv
  - email-triage
  - curriculum-learning
  - reward-hacking
  - trl
  - unsloth
---

# We Trained an LLM to Triage Emails with GRPO

> 📧 **Live environment** · [huggingface.co/spaces/Hk4crprasad/email-triage-env](https://huggingface.co/spaces/Hk4crprasad/email-triage-env)
> 🎮 **Try it now** · [`/demo`](https://hk4crprasad-email-triage-env.hf.space/demo) — Gradio UI on the same Space, with a side-by-side **Baseline vs Trained adapter** tab
> 🤗 **Trained adapter** · [Hk4crprasad/email-triage-grpo](https://huggingface.co/Hk4crprasad/email-triage-grpo) (LoRA on Qwen2.5-3B-Instruct, ~43 MB)
> 💻 **GitHub** · [hk4crprasad/my-env](https://github.com/hk4crprasad/my-env)
> 🏆 **Submission for** · Scaler × Meta PyTorch OpenEnv Hackathon 2026 — Grand Finale, Bangalore

---

## TL;DR

A flooded inbox is one of the most common reasoning tasks a knowledge worker faces, and small LLMs are surprisingly bad at it zero-shot. We built an **OpenEnv-compliant RL environment** with deterministic reward verifiers and trained `Qwen/Qwen2.5-3B-Instruct` with GRPO on a 3-level curriculum. The trained adapter improves the **hard-task final score from 0.29 to 0.59** — a +0.30 gain on the toughest task. The environment is live, the adapter is on HF Hub, and you can run a side-by-side comparison in your browser right now.

---

## Why email triage as an RL task?

Triage is a perfect RL benchmark for personal-task reasoning:

1. **Crisp verification.** Every email has a ground-truth `(category, priority, department, expected_keywords, requires_response)` tuple. We never call an LLM to score; reward is pure deterministic Python.
2. **Multi-decision coordination.** A classifier picks one label. Our agent makes **five coordinated decisions per email** under a step budget. Routing a billing dispute to engineering still gets full classification credit if you guessed `billing` — but you lose the routing reward. Each decision is graded *independently*.
3. **Hidden ambiguity.** "Unauthorised charge" is a billing complaint *and* a possible security incident. A subject of `URGENT: System Down` may turn out to be a quarterly DR drill in the body. The agent has to actually read.
4. **Real-world relevance.** Every developer, manager, and support engineer triages dozens of emails a day. If we can train a 3B model to do it well, that's a small, locally-runnable assistant for a real workflow — not a benchmark trick.

---

## The environment

Built on [OpenEnv](https://github.com/meta-pytorch/OpenEnv) — Meta's standard RL-environment spec. The same WebSocket protocol any TRL/Unsloth/ART/Oumi trainer connects to:

```python
from client import EmailTriageClient
from models import EmailAction

async with EmailTriageClient(base_url="https://hk4crprasad-email-triage-env.hf.space") as env:
    result = await env.reset(task_id="hard", seed=42)
    action = EmailAction(
        email_id="abc123",
        category="billing",
        priority=2,
        department="billing",
        response_draft="We've received your invoice dispute and will reply within 24 hours.",
        escalate=False,
    )
    result = await env.step(action)
    print(result.reward)              # → e.g. +0.55
    print(result.observation.action_feedback)
```

Three difficulty levels build a natural curriculum:

| Task   | Emails | Dimensions agent must produce        | Max steps |
|--------|-------:|--------------------------------------|----------:|
| `easy` | 5      | Classification only                  | 10        |
| `medium` | 10   | + priority + routing                 | 25        |
| `hard` | 20     | + response draft + escalation flag · thread chains, red herrings, GDPR audits, phishing | 40 |

---

## The core innovation: 8 independent reward verifiers

Most RL environments have a single scalar reward. That's a mistake — a clever agent finds a way to game it. We use **8 independent components**, each a pure verifiable function:

| Component | Max | Min | What it measures |
|-----------|----:|----:|------------------|
| `format_compliance` | +0.05 | **−0.15** | Valid JSON, valid `email_id` (gate; runs first) |
| `anti_reprocessing` | 0     | **−0.15** | Re-submitting same email short-circuits other rewards |
| `classification` | +0.30 | −0.15 | Semantic distance: exact +0.30 · adjacent +0.10 · 2-away 0 · 3+ −0.08 |
| `priority`   | +0.20 | −0.10 | Graduated: exact +0.20 · off-by-1 +0.08 · off-by-2 0 · 3+ −0.08 |
| `routing`    | +0.20 | −0.15 | Semantic department distance: exact +0.20 · adjacent 0 · far −0.15 |
| `response_quality` | +0.35 | −0.15 | Hidden-keyword coverage with length sanity check |
| `escalation` | +0.10 | −0.10 | F1-style: TP +0.10 · TN +0.03 · FP −0.10 · FN −0.05 |
| `inbox_completion` | +0.05 | 0 | Bonus for processing every email in one pass |

Five structural anti-hacking measures fall out of the design:

1. **Format gate runs first** — a garbage action gets −0.15 and *nothing else fires*. You can't compensate with a lucky category guess.
2. **Hidden keyword sets** — response quality is graded against keywords the agent never sees. It can't reverse-engineer what to write.
3. **Non-linear priority** — off-by-2+ gets zero (not partial credit). No gradient to exploit.
4. **Re-processing penalty** — flat −0.15 with no other reward applied; can't farm the same email.
5. **Escalation independent of routing** — routing to `management` does *not* give escalation points. Both verifiers fire separately.

`POST /rubric` returns this structure live — judges can verify everything in our blog matches what the server reports.

The verifier-spread chart shows what this looks like empirically:

![Reward Spread](https://huggingface.co/spaces/Hk4crprasad/email-triage-env/resolve/main/plots/reward_spread.png)

*Per-component mean reward over the 20-email hard task. Perfect actions (green) hit the maximum on every verifier; random actions (red) hover near zero. Independence across components is what prevents reward hacking.*

---

## Curriculum: start easy, fail less

If the model never gets reward, it never learns. On `hard` (20 emails, 5 dimensions, response drafts) a cold Qwen2.5-3B-Instruct gets near-zero reward — training stalls because there are no successful trajectories to reinforce.

Three stages, advancement thresholds checked at the end of each:

```
easy  → medium  → hard
  ↑        ↑       ↑
5 emails  10 emails  20 emails
1 dim     3 dims     5 dims
↳ avg reward > 0.60   ↳ avg reward > 0.50
```

The training curve shows the model crossing those thresholds:

![Training Curve](https://huggingface.co/spaces/Hk4crprasad/email-triage-env/resolve/main/plots/training_curve.png)

*Loss decreases and mean reward climbs past the 0.60 advancement threshold within ~200 steps on the easy task — when this threshold is crossed, the curriculum advances to medium.*

---

## Training setup

- **Base model**: `Qwen/Qwen2.5-3B-Instruct`
- **Trainer**: HuggingFace TRL `GRPOTrainer` (no value model — group-relative advantage)
- **Acceleration**: Unsloth (4-bit QLoRA, 2× faster than vanilla HF)
- **LoRA**: r=32, α=64, dropout=0.05, all-linear targets, ~43 MB safetensors
- **Sampling**: G=4 generations per prompt, T=1.0, top-p=0.95
- **Optimization**: lr=3e-6, max_grad_norm=0.1, warmup=20 steps
- **Compute**: free Colab T4 — full curriculum (400 steps total) takes ≈ 45 minutes

The training notebook ([`notebooks/train_grpo.ipynb`](https://github.com/hk4crprasad/my-env/blob/main/notebooks/train_grpo.ipynb)) is end-to-end runnable: it imports `REWARD_FUNCTIONS` directly from the production `train.py` (single source of truth — train and serve use byte-identical reward code), evaluates the model on a held-out seed (99) **before and after** training, generates the comparison plots, and pushes the adapter to HF Hub. A reviewer can re-run it on a free Colab T4 and reproduce the numbers.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hk4crprasad/my-env/blob/main/notebooks/train_grpo.ipynb)

---

## Results: before vs. after GRPO

![Score Comparison](https://huggingface.co/spaces/Hk4crprasad/email-triage-env/resolve/main/plots/score_comparison.png)

| Task   | Baseline (0-shot) | After GRPO | Δ |
|--------|:-----------------:|:----------:|:----:|
| easy   | 0.60 | **0.80** | **+0.20** |
| medium | 0.38 | **0.61** | **+0.23** |
| hard   | 0.29 | **0.59** | **+0.30** |

The improvement *grows* with task difficulty — the model is genuinely learning the policy, not memorising the easy emails.

### Per-dimension breakdown (medium task)

![Dimension Breakdown](https://huggingface.co/spaces/Hk4crprasad/email-triage-env/resolve/main/plots/dimension_breakdown.png)

Routing improves the most. GRPO learns the department-specific signals zero-shot models miss — `XSS` → engineering, `GDPR` → management, `refund` → billing, `multi-language account lockout` → support.

---

## Try the live demo

The HF Space mounts a Gradio UI at [`/demo`](https://hk4crprasad-email-triage-env.hf.space/demo) on the same port as the OpenEnv API. Two tabs:

1. **🎮 Triage one email** — manual play. Pick an action, see the live per-step reward feedback the trainer sees.
2. **🆚 Baseline vs Trained adapter** — load `Qwen/Qwen2.5-3B-Instruct` once, toggle the LoRA adapter on/off via `enable_adapter_layers()` / `disable_adapter_layers()`. Both runs see the exact same email, both actions are scored by the same live verifier, and the ground truth is shown next to them. The Δ-reward pops out instantly.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│             HF Space (Docker, GPU tier)             │
│                                                     │
│  uvicorn server.app:app  (port 7860)                │
│    ├── POST /reset    → new episode                 │
│    ├── POST /step     → action + reward             │
│    ├── GET  /rubric   → all 8 reward components     │
│    ├── GET  /demo     → Gradio UI (mounted)         │
│    └── WS   /ws       → openenv-core protocol       │
│                                                     │
│  EmailTriageEnvironment                             │
│    ├── email_generator.py  (seeded, procedural)     │
│    ├── reward.py           (8 independent funcs)    │
│    ├── graders.py          (episode-level scoring)  │
│    └── tasks.py            (easy/medium/hard)       │
└─────────────────────────────────────────────────────┘
         ↕ WebSocket (openenv-core protocol)
┌─────────────────────────────────────────────────────┐
│            TRL GRPOTrainer (Colab T4)               │
│   + Unsloth (4-bit QLoRA, faster rollouts)          │
│   + EmailTriageClient (subclasses EnvClient)        │
│                                                     │
│   Trains: Qwen/Qwen2.5-3B-Instruct                  │
│   Saves:  Hk4crprasad/email-triage-grpo (LoRA)      │
└─────────────────────────────────────────────────────┘
```

The clean separation — environment handles world dynamics, trainer handles optimisation, model just learns to act — is what keeps the project modular. Other teams can train *their* models against our environment, and we can train *our* model against any other OpenEnv environment, without code changes.

---

## Key design decisions

**Why not a single combined reward?** A weighted-sum like `0.3·class + 0.2·priority` is gameable. A model that always picks `category=billing` on billing emails gets partial reward on everything else for free. Independent verifiers prevent that — there's no free lunch from optimising one dimension.

**Why curriculum learning?** Cold-starting on `hard` gives near-zero reward probability and no gradient. Starting on `easy` bootstraps the policy, and the curriculum advances automatically once average reward crosses 0.60 / 0.50.

**Why seeded generation?** Every episode is reproducible given a `seed`. Judges run `seed=42` and get the exact same inbox every time. Evaluation is fair, results are comparable across runs.

**Why GRPO over PPO?** GRPO drops the value model. For our task — which has crisp verifiable rewards — we don't need a learned value estimate. GRPO computes advantage from a group of N=4 completions per prompt. Less memory, simpler setup, same or better performance for verifiable-reward tasks.

**Why a 3B model?** It fits on a free Colab T4. The point isn't that a 70B model can triage emails — it's that a 3B model can be *trained* to do it significantly better using an RL environment. The architecture scales to any model; we chose the smallest one that demonstrates learning.

---

## Run it yourself

**1-line API call:**

```bash
curl -X POST https://hk4crprasad-email-triage-env.hf.space/reset \
  -H 'Content-Type: application/json' \
  -d '{"task_id":"easy","seed":42}'
```

**Full episode in 60 seconds:**

```python
import asyncio
from client import EmailTriageClient
from models import EmailAction

async def demo():
    async with EmailTriageClient('https://hk4crprasad-email-triage-env.hf.space') as env:
        r = await env.reset(task_id='easy', seed=42)
        email = r.observation.emails[0]
        action = EmailAction(
            email_id=email['email_id'],
            category='billing',
            priority=2,
            department='billing',
        )
        r = await env.step(action)
        print(f'Reward: {r.reward:+.3f}')
        print(f'Feedback: {r.observation.metadata.get("action_feedback", "")}')

asyncio.run(demo())
```

**Run the trained adapter locally:**

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct', device_map='auto')
model = PeftModel.from_pretrained(base, 'Hk4crprasad/email-triage-grpo')
tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
```

**Run the inference benchmark with the trained model on all 3 tasks:**

```bash
export HF_TOKEN="hf_..."
export USE_LOCAL_MODEL=1
python inference.py
```

---

## What's next

- Scale to `Qwen2.5-7B-Instruct` and `Llama-3.1-8B-Instruct` and benchmark gain at parameters
- Multi-turn thread context (agents see full reply chains, not just the latest email)
- Human preference alignment layer on top of GRPO
- Production simulation: 200 emails/day, latency budget, concurrent users

The environment is live, the code is open, the trained adapter is on HF Hub. We'd love to see what others train on it.

---

## Links

| Resource | Link |
|----------|------|
| 🤗 HF Space (live API + Gradio demo) | [spaces/Hk4crprasad/email-triage-env](https://huggingface.co/spaces/Hk4crprasad/email-triage-env) |
| 🎮 Live Gradio UI | [/demo](https://hk4crprasad-email-triage-env.hf.space/demo) |
| 🧠 Trained LoRA adapter | [Hk4crprasad/email-triage-grpo](https://huggingface.co/Hk4crprasad/email-triage-grpo) |
| 💻 GitHub | [hk4crprasad/my-env](https://github.com/hk4crprasad/my-env) |
| 🎓 Training Colab | [train_grpo.ipynb](https://colab.research.google.com/github/hk4crprasad/my-env/blob/main/notebooks/train_grpo.ipynb) |
| 🧪 Demo Colab | [demo_and_test.ipynb](https://colab.research.google.com/github/hk4crprasad/my-env/blob/main/notebooks/demo_and_test.ipynb) |
| 📊 Reward rubric (live JSON) | `GET /rubric` |
| 🧭 Curriculum schedule (live JSON) | `GET /curriculum` |

---

*Built by Team Ctrl-Alt-Defeat for the Scaler × Meta PyTorch OpenEnv Hackathon 2026.*
*Questions? Open an issue on GitHub or ping us on the hackathon Discord.*
