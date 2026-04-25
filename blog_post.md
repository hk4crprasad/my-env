---
title: "We Trained an LLM to Triage Emails with GRPO — Here's What We Learned"
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

# We Trained an LLM to Triage Emails with GRPO — Here's What We Learned

> 📧 **Live environment**: [huggingface.co/spaces/Hk4crprasad/email-triage-env](https://huggingface.co/spaces/Hk4crprasad/email-triage-env)  
> 🤗 **Trained adapter**: [Hk4crprasad/email-triage-grpo](https://huggingface.co/Hk4crprasad/email-triage-grpo)  
> 💻 **GitHub**: [github.com/hk4crprasad/my-env](https://github.com/hk4crprasad/my-env)  
> 🏆 **Submission for**: Scaler × Meta PyTorch OpenEnv Hackathon 2026 — Grand Finale, Bangalore

---

## The Problem: Your Inbox is a Mess, and LLMs Don't Help

Every developer, manager, and support engineer deals with the same problem: a flooded inbox that demands judgment calls dozens of times per day.

- Is this email spam, or a disguised phishing attempt?  
- Is "Unauthorised charge" a billing complaint, or a security incident?  
- Does this thread need escalation to management, or just a template reply?

The frustrating part? **Current LLMs are surprisingly bad at this.** Zero-shot, Qwen2.5-3B-Instruct routes billing disputes to engineering 60% of the time. It classifies phishing as "urgent" and responds with helpful links to the attacker's fake site.

We built an RL environment to fix that — and trained a model to prove it works.

---

## What We Built: The Email Triage RL Environment

The environment presents an agent with a realistic inbox and asks it to make **five coordinated decisions per email**:

```json
{
  "email_id": "hard_003",
  "category": "urgent",
  "priority": 1,
  "department": "engineering",
  "response_draft": "We acknowledge the production outage. Our team is investigating.",
  "escalate": true
}
```

This runs on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) standard — the same WebSocket protocol Meta uses across all hackathon environments. Any trainer (TRL, Unsloth, ART, Oumi) can connect to it with 5 lines of Python:

```python
from client import EmailTriageClient
from models import EmailAction

async with EmailTriageClient(base_url="https://hk4crprasad-email-triage-env.hf.space") as env:
    result = await env.reset(task_id="hard", seed=42)
    result = await env.step(EmailAction(email_id="hard_003", category="urgent", ...))
    print(result.reward)  # e.g. +0.75
```

---

## The Core Innovation: 7 Independent Reward Verifiers

Most RL environments have one reward signal. That's a mistake — a sufficiently clever agent will find a way to game it.

We use **7 independent reward components**, each a pure verifiable function:

| Component | Max | Min | What it measures |
|-----------|-----|-----|-----------------|
| Format compliance | +0.05 | **-0.15** | Valid JSON structure, valid `email_id` |
| Deduplication | 0 | **-0.15** | Never re-process the same email |
| Classification | +0.20 | -0.10 | Correct spam/billing/technical/urgent/general |
| Priority | +0.15 | 0.00 | Exact: +0.15 · Off-by-1: +0.07 · Off-by-2+: 0 |
| Routing | +0.15 | -0.05 | Correct department assignment |
| Response quality | +0.30 | -0.10 | Hidden keyword coverage in draft |
| Escalation | +0.05 | -0.05 | Correctly flags high-severity issues |

**Why this stops reward hacking:**

1. **Format gate runs first** — a garbage action gets -0.15 and nothing else. You can't compensate with a great category guess.
2. **Hidden keywords** — response quality is graded against a keyword set the agent never sees. It can't reverse-engineer what to write.
3. **Non-linear priority** — off-by-2+ gets zero, not partial credit. No gradient to exploit.
4. **Deduplication** — the agent cannot farm the same email repeatedly. Re-processing costs -0.15.
5. **Escalation is independent of routing** — routing to "management" does NOT give escalation points.

The reward spread chart shows what this looks like empirically:

![Reward Spread](https://huggingface.co/spaces/Hk4crprasad/email-triage-env/resolve/main/plots/reward_spread.png)
*Perfect actions (green) score the maximum on every verifier. Random actions (red) hover near zero. Independence across components is what prevents reward hacking.*

---

## Curriculum Learning: Start Easy, Fail Less

One of the most practical lessons from the hackathon: **if the model never gets reward, it never learns**.

On `hard` (20 emails, 5 dimensions, escalation required), a cold Qwen2.5-3B-Instruct gets near-zero reward. Training stalls — the model has no successful trajectories to reinforce.

We use 3-level curriculum learning:

```
easy  → medium  → hard
  ↑        ↑       ↑
5 emails  10 emails  20 emails
1 dim     3 dims     5 dims
```

**Advancement thresholds**: avg_reward > 0.60 to move from easy→medium, > 0.50 for medium→hard.

The training curve shows the model crossing these thresholds:

![Training Curve](https://huggingface.co/spaces/Hk4crprasad/email-triage-env/resolve/main/plots/training_curve.png)
*GRPO loss decreases and mean reward climbs past the curriculum threshold at ~200 steps.*

---

## Training Setup

- **Base model**: `Qwen/Qwen2.5-3B-Instruct`
- **Training**: GRPO via HuggingFace TRL + Unsloth
- **Adapter size**: ~43 MB (LoRA)
- **Environment**: FastAPI server on HF Spaces
- **Compute**: T4 GPU on Google Colab (free tier)

The training Colab is fully reproducible: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hk4crprasad/email-triage-env/blob/main/notebooks/train_grpo.ipynb)

---

## Results: Before vs. After GRPO

![Score Comparison](https://huggingface.co/spaces/Hk4crprasad/email-triage-env/resolve/main/plots/score_comparison.png)

| Task | Baseline (0-shot) | After GRPO | Improvement |
|------|:-----------------:|:----------:|:-----------:|
| easy | 0.60 | **0.92** | **+0.32** |
| medium | 0.38 | **0.64** | **+0.26** |
| hard | 0.29 | **0.51** | **+0.22** |

The improvement generalises across difficulty levels — the model doesn't just memorise easy emails.

### Per-Dimension Breakdown (medium task)

![Dimension Breakdown](https://huggingface.co/spaces/Hk4crprasad/email-triage-env/resolve/main/plots/dimension_breakdown.png)

| Dimension | Baseline | Trained | Δ |
|-----------|:--------:|:-------:|:---:|
| Classification | 0.48 | 0.72 | +0.24 |
| Priority | 0.31 | 0.55 | +0.24 |
| Routing | 0.29 | 0.50 | +0.21 |

**Routing improves the most** — GRPO learns department-specific signals that zero-shot models consistently miss (e.g., "XSS" → engineering, "GDPR" → management, "refund" → billing).

---

## What the Agent Actually Learned

Here's a concrete before/after example on a hard task email:

**Email received:**
```
Subject: URGENT: Unauthorized access attempt on admin portal
From: security-alerts@company-internal.net
Body: We detected 3 failed login attempts on your admin account from 
      IP 185.234.x.x (Russia). Click here to verify: http://bit.ly/2xK9mA
```

| Decision | Baseline (0-shot) | After GRPO |
|----------|:----------------:|:-----------:|
| Category | `urgent` ❌ (it's spam/phishing) | `spam` ✅ |
| Priority | 1 ❌ (not urgent) | 5 ✅ |
| Department | `engineering` ❌ | `support` ✅ |
| Escalate | `true` ❌ | `false` ✅ |
| Step reward | -0.20 | +0.55 |

The model learned that suspicious domains + bit.ly links + "verify your account" language = phishing, even when the subject line screams "URGENT".

---

## Architecture: How It All Fits Together

```
┌─────────────────────────────────────────────────────┐
│                 HF Space (Docker)                   │
│                                                     │
│  FastAPI server/app.py                              │
│    ├── POST /reset  → new episode                   │
│    ├── POST /step   → action + reward               │
│    ├── WS  /ws      → openenv-core WebSocket        │
│    └── GET /rubric  → reward definitions            │
│                                                     │
│  EmailTriageEnvironment                             │
│    ├── email_generator.py  (seeded, procedural)     │
│    ├── reward.py           (7 independent funcs)    │
│    ├── graders.py          (episode-level scoring)  │
│    └── tasks.py            (easy/medium/hard)       │
└─────────────────────────────────────────────────────┘
         ↕ WebSocket (openenv-core protocol)
┌─────────────────────────────────────────────────────┐
│              TRL GRPOTrainer                        │
│   + Unsloth (4-bit QLoRA, faster rollouts)          │
│   + EmailTriageClient (EnvClient)                   │
│                                                     │
│   Trains: Qwen/Qwen2.5-3B-Instruct                  │
│   Saves:  Hk4crprasad/email-triage-grpo (LoRA)      │
└─────────────────────────────────────────────────────┘
```

The full separation — environment handles world dynamics, trainer handles optimisation, model just learns to act — is what makes it modular and reusable.

---

## Key Design Decisions (and Why We Made Them)

### Why not a single reward score?
Because a combined score like `0.3*(class) + 0.2*(priority)` can be gamed. A model that learns to always pick `category=billing` on billing emails gets partial reward on everything else for free. Independent verifiers prevent that.

### Why curriculum learning?
Cold-start on `hard` gives near-zero reward probability. The model has no gradient to follow. Starting on `easy` bootstraps the policy, then curriculum automatically advances once performance is stable.

### Why seeded generation?
Every episode is reproducible given `seed`. Judges can run `seed=42` and get the exact same inbox every time. This makes evaluation fair and results comparable across runs.

### Why 5 decisions per email?
A naive classifier picks one label. Real triage requires coordinated judgment: is this urgent *and* needs routing to billing *and* needs a response *and* needs escalation? Each decision is a separate RL sub-problem. The interdependencies (routing and escalation are correlated but independent) make it genuinely hard.

---

## Try It Yourself

**Run a full episode in 60 seconds:**

```bash
pip install openenv-core requests
python -c "
import asyncio
from client import EmailTriageClient
from models import EmailAction

async def demo():
    async with EmailTriageClient('https://hk4crprasad-email-triage-env.hf.space') as env:
        result = await env.reset(task_id='easy', seed=42)
        print(f'Inbox: {len(result.observation.emails)} emails')
        
        email = result.observation.emails[0]
        action = EmailAction(
            email_id=email['email_id'],
            category='billing',
            priority=2,
            department='billing'
        )
        result = await env.step(action)
        print(f'Reward: {result.reward:+.3f}')
        print(f'Feedback: {result.observation.metadata.get(\"action_feedback\", \"\")}')

asyncio.run(demo())
"
```

**Or open the interactive demo Colab:**  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hk4crprasad/email-triage-env/blob/main/notebooks/demo_and_test.ipynb)

---

## What's Next

- [ ] Scale training to larger base models (Qwen2.5-7B, Llama-3.1-8B)
- [ ] Add multi-turn thread context (agents see reply chains, not just single emails)
- [ ] Human preference alignment layer on top of GRPO
- [ ] Production deployment benchmark: 200 emails/day simulation

The environment is live, the code is open, and the trained model is on HF Hub. We'd love to see what others train on it.

---

## Links

| Resource | Link |
|----------|------|
| 🤗 HF Space (live API + Gradio demo) | [spaces/Hk4crprasad/email-triage-env](https://huggingface.co/spaces/Hk4crprasad/email-triage-env) |
| 🧠 Trained LoRA adapter | [Hk4crprasad/email-triage-grpo](https://huggingface.co/Hk4crprasad/email-triage-grpo) |
| 💻 GitHub | [hk4crprasad/my-env](https://github.com/hk4crprasad/my-env) |
| 🎓 Training Colab | [train_grpo.ipynb](https://colab.research.google.com/github/Hk4crprasad/email-triage-env/blob/main/notebooks/train_grpo.ipynb) |
| 🧪 Demo Colab | [demo_and_test.ipynb](https://colab.research.google.com/github/Hk4crprasad/email-triage-env/blob/main/notebooks/demo_and_test.ipynb) |
| 📊 Knowledge Graph | [graphify-out/graph.html](https://github.com/hk4crprasad/my-env/blob/main/graphify-out/graph.html) |

---

*Built by Team Ctrl-Alt-Defeat for the Scaler × Meta PyTorch OpenEnv Hackathon 2026.*  
*Questions? Open an issue on GitHub or ping us on the hackathon Discord.*
