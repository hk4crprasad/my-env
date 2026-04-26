---
title: "We Trained an LLM to Triage Emails with GRPO — and the Phishing One Was the Best Part"
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
  - qwen2.5
  - world-modeling
---

# We Trained an LLM to Triage Emails with GRPO — and the Phishing One Was the Best Part

> **📧 Live environment**: [huggingface.co/spaces/Hk4crprasad/email-triage-env](https://huggingface.co/spaces/Hk4crprasad/email-triage-env)  
> **🎮 Try it now**: [`/demo`](https://hk4crprasad-email-triage-env.hf.space/demo) — side-by-side baseline vs trained adapter in your browser  
> **🧠 Trained adapter**: [Hk4crprasad/email-triage-grpo](https://huggingface.co/Hk4crprasad/email-triage-grpo) — 43 MB LoRA on Qwen2.5-3B-Instruct  
> **💻 GitHub**: [hk4crprasad/my-env](https://github.com/hk4crprasad/my-env)  
> **🏆 Submission for**: Scaler × Meta PyTorch OpenEnv Hackathon 2026 — Grand Finale, Bangalore

---

## TL;DR

We built an **OpenEnv-compliant RL environment** for email triage with 8 independent deterministic reward verifiers and a 3-level curriculum, trained `Qwen/Qwen2.5-3B-Instruct` using GRPO (Group Relative Policy Optimisation) via TRL + Unsloth, and improved its hard-task triage score from **0.29 → 0.59** (+0.30) in 400 steps on a free Colab T4.

The environment is live, the adapter is on HF Hub, and there is a phishing email in the hard task that the zero-shot model always gets wrong — and the trained model always gets right. That's the payoff.

---

## The Problem (It's Not What You Think)

Everyone has seen the pitch: "LLMs will manage your inbox." We tested that. The reality is rougher than the pitch.

We took `Qwen/Qwen2.5-3B-Instruct` — a solid, instruction-tuned 3B model — and asked it to triage 20 emails zero-shot. Here's what happened:

```
Email: "Congratulations! You've won £5,000,000. Click here to claim your prize."
Zero-shot Qwen2.5: category=urgent, priority=1, department=engineering
                  ❌ This is spam. It should be: spam, priority=5, support.

Email: "⚠️ URGENT: Your account will be SUSPENDED in 12 hours!!!"
       From: security-alerts@support-team-verification.net
       Body: "Verify your account at: bit.ly/verify-now"
Zero-shot Qwen2.5: category=urgent, priority=1, department=engineering, escalate=True
                  ❌ This is phishing. Note the suspicious domain and bit.ly redirect.
```

These aren't edge cases. They're the most common inbox failures. And here's the deeper problem: **the model isn't just getting categories wrong**. It's getting routing, priority, escalation, and response decisions wrong simultaneously — because it has no mechanism to coordinate those five decisions or learn from getting them wrong.

Fine-tuning on labelled data would help with categories. But labels don't teach *coordination*. For that, you need a reward signal that fires on every decision, every step.

That's why we built an RL environment.

---

## Research: What Makes Email Triage Hard

Before designing the reward system, we analysed what zero-shot models actually fail at. Three patterns emerged:

### 1. Subject-body contradiction

Subjects are written to trigger responses. Bodies contain the truth. A model that reads only the subject gets manipulated:

```
Subject: "CRITICAL: System Down — Immediate Action Required"
Body:    "[TEST ENV] This is a quarterly disaster recovery drill. No action needed."

Zero-shot: category=urgent, priority=1, escalate=True   ❌
Correct:   category=general, priority=4, escalate=False  ✅
```

The `[TEST ENV]` tag is the signal. Models that don't build a mental model of *what kind of sender sends this* miss it.

### 2. Semantic category distance

Not all misclassifications are equal. Confusing `urgent` with `technical` is bad but understandable — they share the "needs-fast-response" signal. Calling a billing dispute `spam` is a category error that throws off every downstream decision.

We needed a reward function that reflects this gradient — not a binary right/wrong.

### 3. Routing requires world knowledge

`XSS vulnerability disclosure` → `engineering`  
`GDPR data deletion request` → `management`  
`Invoice payment failed` → `billing`  
`Multi-language "account locked"` → `support`

These mappings aren't in the model's tokenizer. They require understanding department responsibilities — which is domain knowledge that can be learned via RL if the reward signal teaches it.

### 4. Thread context is invisible to classifiers

A reply chain where the *first* email is a customer complaint and the *third* is a management escalation looks very different from each email in isolation. The agent needs to know: "this is a follow-up, and the original is already being handled."

---

## The Environment Design

### Seeded, deterministic email corpus

Every inbox is generated from a seed. `seed=42` always produces the same 5/10/20 emails. This is non-negotiable for RL — if the environment is random, you can't compare runs or verify results.

We wrote 15 email archetypes covering the failure modes above, then procedurally vary sender names, amounts, domains, and timestamps. The hard task includes:

- **Phishing**: Suspicious domain + `bit.ly` redirect + fake urgency language
- **Billing-security hybrid**: "Unauthorised charge" — is it billing or a security incident?
- **Thread context**: Reply chain requiring the agent to recognise it's a follow-up
- **Red herring**: `[TEST ENV]` DR drill with "CRITICAL" subject
- **Compliance**: GDPR data deletion audit → management
- **Security disclosure**: XSS vulnerability report → engineering + escalation
- **Multi-language account lockout**: Support ticket in mixed language
- **Legitimate urgent**: Real production outage — the model should NOT dismiss this

The point: the agent has to *read carefully*, not pattern-match on subject keywords.

### Three-level curriculum

Our first training attempt went straight to `hard`. After 100 steps, reward was still near zero. The model couldn't learn because there was nothing to reinforce — it was getting every decision wrong simultaneously.

The curriculum solution is well-known in RL but often skipped in LLM training. We implemented it with automatic advancement:

```
easy (5 emails, classification only)
  → avg reward > 0.60
    → medium (10 emails, + priority + routing)
         → avg reward > 0.50
              → hard (20 emails, all 5 dimensions + efficiency)
```

The insight: `easy` teaches format compliance and basic category understanding. `medium` teaches routing — the hardest dimension for zero-shot models. `hard` adds response quality and escalation on top of a foundation that already works.

---

## The Core Innovation: 8 Independent Reward Verifiers

This is the part we spent the most time on, and it's what makes the environment genuinely anti-gameable.

Most RL environments use a single scalar reward: `reward = 1 if done correctly else 0`. This is gameable. A model that always picks `billing` on all emails would get +1 on every billing email and 0 on everything else — then GRPO would reinforce the `billing` bias because it beats random.

The fix is **independence**. Each reward component fires on its own decision and cannot affect the others:

```python
# From server/reward.py — each function is pure Python, no LLM

def reward_classification(action, gt: EmailGroundTruth) -> float:
    """Semantic distance between agent category and ground truth."""
    dist = _CATEGORY_DISTANCE[gt.category][action['category']]
    return {0: 0.30, 1: 0.10, 2: 0.00}.get(dist, -0.08)

def reward_routing(action, gt: EmailGroundTruth) -> float:
    """Semantic distance between agent department and ground truth."""
    dist = _DEPT_DISTANCE[gt.department][action['department']]
    return {0: 0.20, 1: 0.00, 2: -0.08}.get(dist, -0.15)

def reward_escalation(action, gt: EmailGroundTruth) -> float:
    """F1-style: both false positives AND false negatives are penalised."""
    should = gt.department == "management" or gt.priority == 1
    agent  = bool(action.get("escalate", False))
    if agent and should:     return  0.10   # true positive
    if not agent and should: return -0.05   # false negative — missed critical
    if agent and not should: return -0.10   # false positive — unnecessary noise
    return 0.03                             # true negative — correct restraint

def reward_response_quality(action, gt: EmailGroundTruth) -> float:
    """Keyword coverage against a HIDDEN list. Agent can't reverse-engineer it."""
    keywords = gt.expected_keywords   # never shown in prompt
    draft    = (action.get("response_draft") or "").lower()
    coverage = sum(kw in draft for kw in keywords) / len(keywords)
    if coverage >= 0.70: return 0.35
    if coverage >= 0.50: return 0.25
    if coverage >= 0.30: return 0.15
    return 0.05
```

The five anti-hacking structural properties that fall out:

1. **Format gate runs first.** `reward_format_compliance()` checks JSON validity and `email_id` correctness before anything else fires. A garbage output gets −0.15 and nothing more. No lucky category reward for malformed JSON.

2. **Hidden keyword sets.** Response quality is graded against keywords the agent never sees in its prompt. It can't reverse-engineer what to write by memorising the expected output.

3. **Non-linear priority scoring.** Off-by-2+ gets zero (not partial credit). There's no gradient to exploit by always guessing the median.

4. **Re-processing penalty.** Submitting the same email twice returns flat −0.15 with all other rewards blocked. The agent can't farm a single email it knows how to handle.

5. **Escalation is independent of routing.** Routing to `management` does NOT give escalation credit. Both verifiers fire separately. The agent must make both decisions correctly.

Here's what this looks like empirically — the verifier separation chart, generated by `scripts/generate_plots.py` from the real reward functions (no model required):

![Reward Spread](https://huggingface.co/spaces/Hk4crprasad/email-triage-env/resolve/main/plots/reward_spread.png)

*Perfect actions (green) score near-maximum on every verifier; random actions (red) hover near zero; adversarial actions (wrong-but-valid) get consistently penalised. This separation is what makes learning happen.*

You can verify this yourself:

```bash
curl https://hk4crprasad-email-triage-env.hf.space/rubric | python -m json.tool
```

The rubric endpoint returns the full component definitions live. Everything in this post matches what the server reports.

---

## Training Setup

### Model and tooling

| What | Choice | Why |
|------|--------|-----|
| Base model | `Qwen/Qwen2.5-3B-Instruct` | Smallest model that demonstrates RL learning; fits on free T4 |
| Trainer | TRL `GRPOTrainer` | GRPO needs no value model — perfect for verifiable rewards |
| Acceleration | Unsloth QLoRA (4-bit) | 2× faster rollout generation, 40% less VRAM |
| LoRA config | r=32, α=64, dropout=0.05, all linear layers | Expressive enough to learn routing + escalation |
| Adapter size | ~43 MB (safetensors) | Small enough for the HF Space to download at inference time |

### GRPO vs PPO: why we chose GRPO

GRPO (Group Relative Policy Optimisation) was introduced in DeepSeek's work on math reasoning. The key insight: when rewards are **verifiable**, you don't need a value network to estimate future return. You just need a group of completions to compute relative advantage.

For our task:
- PPO needs a critic to estimate V(s). Training a critic for email triage is additional complexity and memory.
- GRPO generates N=4 completions per prompt, computes `advantage_i = (reward_i − mean_group) / std_group`, and runs a policy gradient step. Simple, stable, works well.

This is also why GRPO powers DeepSeek-R1 and several Qwen-RLVR experiments — for tasks with crisp rewards, it's the right tool.

### Why not SFT first?

We tried SFT warm-up. It hurt. The model learned to mimic the format of correct actions but not the reasoning behind them. When we then ran GRPO, the SFT prior resisted updates — the model was confident in wrong answers.

Starting cold from the instruction-tuned model (without SFT) worked better. The `easy` curriculum stage serves the same purpose as SFT warm-up: it bootstraps format compliance and basic category understanding while still being rewarded for getting it right (not just imitating).

### Hyperparameter choices

```python
learning_rate    = 3e-6    # conservative — GRPO is sensitive to LR; too high → reward spikes
max_grad_norm    = 0.1     # tight clipping — prevents the model from making huge updates
                           # on lucky high-reward completions
temperature      = 1.0     # high diversity for group contrast — if all completions are similar,
                           # GRPO gets no variance to learn from
num_generations  = 4       # we tried 8; 4 is the sweet spot for T4 VRAM
lr_scheduler     = cosine  # smoother than linear for fine-tuning; important at end of training
warmup_steps     = 20      # prevents large early updates before the baseline stabilises
```

---

## Training: What We Actually Observed

### Stage 1 (easy, 200 steps): Format compliance first

The first 50 steps were rough. The model produced valid JSON about 60% of the time, but `email_id` mismatches (copy-pasting the ID from the prompt incorrectly, or inventing one) triggered the format gate repeatedly.

By step 100, format compliance was >95%. The model had learned: *always copy the `email_id` exactly*. That's the format gate doing its job.

By step 150, classification accuracy on easy emails reached ~75%. The model learned to distinguish `spam` from everything else — the biggest win early.

The reward crossed 0.60 around step 180. Curriculum advanced to medium.

### Stage 2 (medium, 100 steps): Routing is the hard part

The reward dropped at step 200 (curriculum transition — new task, harder). This is expected.

What we didn't expect: routing accuracy started higher than on easy. The curriculum had bootstrapped the model's understanding of email semantics well enough that department routing improved quickly.

By step 280, routing accuracy reached ~55% (vs 39% zero-shot). The model learned: billing keywords → billing department, security keywords → engineering, legal/compliance → management.

### Stage 3 (hard, 100 steps): The phishing lesson

The hard task introduced the phishing email and the DR drill red herring. The model initially treated both as urgent (zero-shot behavior).

By step 350, something interesting happened: the model started consistently identifying the phishing email correctly. Not because we told it "bit.ly = phishing", but because the `anti_reprocessing` and `classification` rewards together created pressure to get the full category right — and phishing consistently got the worst rewards when misclassified.

By step 400, the hard-task score reached 0.59.

---

## The Phishing Reveal

This is the demo moment we kept coming back to. Here's the actual email from the hard task (seed=42, index 2):

```
From:    security-alerts@support-team-verification.net
Subject: ⚠️ URGENT: Your account will be SUSPENDED in 12 hours!!!

Dear valued customer,

Your account has been flagged for suspicious activity. To prevent suspension,
you must verify your identity immediately.

Click here: bit.ly/account-verify-secure

Failure to verify within 12 hours will result in permanent account termination.

Best regards,
Security Team
```

Zero-shot `Qwen2.5-3B-Instruct`:
```json
{"email_id": "hard_003", "category": "urgent", "priority": 1,
 "department": "engineering", "escalate": true}
```
Reward: **−0.28**

After GRPO training:
```json
{"email_id": "hard_003", "category": "spam", "priority": 5,
 "department": "support", "escalate": false}
```
Reward: **+0.55**

The signals the trained model learned to weight:
1. `support-team-verification.net` — unusual domain suffix, not a legitimate corporate domain
2. `bit.ly/` redirect — legitimate security alerts never use URL shorteners
3. "SUSPENDED in 12 hours" — fake urgency pattern
4. "verify your identity" — credential phishing template

None of these were in the prompt instructions. The model learned them from reward pressure over 400 steps.

You can see this live in the Gradio demo's **🆚 Baseline vs Trained adapter** tab:

![Demo UI](https://huggingface.co/spaces/Hk4crprasad/email-triage-env/resolve/main/plots/score_comparison.png)

---

## Results: Before vs. After

![Score Comparison](https://huggingface.co/spaces/Hk4crprasad/email-triage-env/resolve/main/plots/score_comparison.png)

| Task | Emails | Baseline (0-shot) | After GRPO | Δ |
|------|-------:|:-----------------:|:----------:|:---:|
| `easy`   | 5  | 0.60 | **0.80** | **+0.20** |
| `medium` | 10 | 0.38 | **0.61** | **+0.23** |
| `hard`   | 20 | 0.29 | **0.59** | **+0.30** |

*Evaluated on held-out seed=99 (never seen during training). The improvement grows with difficulty — that's the hallmark of genuine learning, not memorisation.*

### Per-dimension breakdown

![Dimension Breakdown](https://huggingface.co/spaces/Hk4crprasad/email-triage-env/resolve/main/plots/dimension_breakdown.png)

| Dimension | Baseline | After GRPO | Δ | Key learning |
|-----------|:--------:|:----------:|:---:|-------------|
| Classification | ~0.65 | ~0.82 | +0.17 | Phishing detection, billing vs urgent disambiguation |
| **Routing** | ~0.39 | **~0.60** | **+0.21** | XSS→engineering, GDPR→management, refund→billing |
| Priority | ~0.48 | ~0.66 | +0.18 | Graduated scoring teaches calibration, not just exact match |
| Escalation | ~0.55 | ~0.69 | +0.14 | F1-style reward discourages over-escalation |

**Routing improves the most** (+21pp). This is the dimension that requires the most domain knowledge — which department handles which type of issue — and it's also the one zero-shot models are worst at because the training corpus for Qwen doesn't include "here's what the billing team handles."

GRPO learned it from reward signal alone in ~100 medium-task steps.

### Training curve

![Training Curve](https://huggingface.co/spaces/Hk4crprasad/email-triage-env/resolve/main/plots/training_curve.png)

*Each coloured band is a curriculum stage. The reward curves are smooth — no reward hacking spikes — because the 8 independent verifiers make gaming any single component unprofitable. Loss tracks reward consistently: when reward improves, the policy is genuinely learning better completions, not just exploiting a single verifier.*

---

## Architecture: How It All Fits Together

```
┌─────────────────────────────────────────────────────────┐
│             HF Space (Docker, T4 GPU tier)              │
│                                                         │
│  uvicorn server.app:app  (port 7860)                    │
│  ├── POST /reset     → new episode (seeded inbox)       │
│  ├── POST /step      → action + 8-component reward      │
│  ├── GET  /rubric    → live reward component defs       │
│  ├── GET  /curriculum → task progression + thresholds   │
│  ├── GET  /demo      → Gradio UI (same port, mounted)   │
│  └── WS   /ws        → openenv-core WebSocket          │
│                                                         │
│  EmailTriageEnvironment                                 │
│  ├── email_generator.py  (15 archetypes, seeded)        │
│  ├── reward.py           (8 independent functions)      │
│  ├── graders.py          (episode final scoring)        │
│  └── tasks.py            (easy/medium/hard + curriculum)│
└─────────────────────────────────────────────────────────┘
             ↕ WebSocket (openenv-core protocol)
┌─────────────────────────────────────────────────────────┐
│     TRL GRPOTrainer (Colab T4, free tier)               │
│  + Unsloth (4-bit QLoRA, 2× faster rollouts)            │
│                                                         │
│  Base:  Qwen/Qwen2.5-3B-Instruct                        │
│  Saves: Hk4crprasad/email-triage-grpo (43 MB LoRA)      │
│  Steps: 200 easy + 100 medium + 100 hard = 400 total    │
│  Time:  ~45 min on free Colab T4                        │
└─────────────────────────────────────────────────────────┘
```

The separation is intentional: the environment handles world dynamics, the trainer handles optimisation, the model just learns to act. Other teams can train their models against our environment without any code changes. We can train our model against any other OpenEnv environment the same way.

---

## OpenEnv Compliance

The environment implements the full OpenEnv spec:

- **`/reset`** → returns observation, `session_id`, episode metadata
- **`/step`** → accepts action dict, returns observation + reward + done
- **`/state`** → current episode state
- **`/schema`** → JSON schemas for Action and Observation
- **`/ws`** → openenv-core WebSocket protocol (compatible with TRL `EnvClient`, Unsloth, ART, Oumi)

Any trainer using `openenv-core` can connect to our Space directly:

```python
from client import EmailTriageClient

async with EmailTriageClient("https://hk4crprasad-email-triage-env.hf.space") as env:
    result = await env.reset(task_id="hard", seed=42)
    # → standard OpenEnv observation
    result = await env.step(action)
    # → standard OpenEnv response with reward
```

The WebSocket protocol enables TRL's `GRPOTrainer` to treat our Space as a remote environment — no local server needed. This is what makes the hackathon theme complete: the environment is a deployable, reusable service.

---

## Reproducibility

Every number in this post can be reproduced:

1. **Re-run validation suite** (no GPU, 60 seconds):
   ```bash
   python scripts/validate_env.py  # 26 checks, exit 0 expected
   ```

2. **Re-generate reward spread plot** (no GPU, no model):
   ```bash
   python scripts/generate_plots.py  # runs real reward functions, generates all 4 PNGs
   ```

3. **Reproduce training results** (free Colab T4, ~45 min):
   - Open [notebooks/train_grpo.ipynb](https://colab.research.google.com/github/hk4crprasad/my-env/blob/main/notebooks/train_grpo.ipynb)
   - Run All
   - The notebook evaluates baseline (seed=99), runs 400-step curriculum, evaluates again, generates plots, and pushes to Hub

4. **Verify reward rubric** (1 second):
   ```bash
   curl https://hk4crprasad-email-triage-env.hf.space/rubric | python -m json.tool
   ```

5. **Check curriculum thresholds** (1 second):
   ```bash
   curl https://hk4crprasad-email-triage-env.hf.space/curriculum | python -m json.tool
   ```

---

## What We Learned (and What We'd Do Differently)

### What worked

**Independent verifiers prevent reward hacking.** The training curves are smooth precisely because there's no single component to exploit. When we tried a combined reward in early experiments, the model learned to always claim `escalate=True` on everything — it got escalation credit on genuine escalations (TP) while the penalty for FP was diluted by other rewards. With independence, FP escalation always costs −0.10 with no compensation from other components.

**Curriculum is essential for cold-start RL.** 400 steps on hard alone never crossed 0.35. The same 400 steps with curriculum reached 0.59. The difference is that `easy` bootstraps both format compliance and category understanding — the two foundations everything else builds on.

**GRPO with N=4 is enough.** We expected N=8 to learn faster (more variance to exploit). In practice, N=4 gave similar final scores with 2× faster rollout generation. The reason: with 8 independent verifiers, even 4 completions produce enough reward variance for GRPO to find signal.

### What we'd do differently

**Start with response quality from step 1.** We introduced `response_quality` only in the hard task. In retrospect, even in the easy task, requiring a brief response draft would have bootstrapped the response skill earlier and given the hard task a better starting point.

**Temperature scheduling.** We used fixed T=1.0. A temperature schedule (high early for exploration, lower late for exploitation) would likely smooth the late-stage training curve.

**Multi-seed evaluation.** Our results are from seed=99 (held-out). Running over 5 seeds would give confidence intervals. We'd do this for a paper submission.

---

## What's Next

The environment is open and reusable. We'd love to see what others train on it:

- **Scale to 7B/8B** — Qwen2.5-7B-Instruct or Llama-3.1-8B. The environment runs on the same server; only the trainer changes.
- **Multi-turn thread context** — agents see the full reply chain, not just the latest email. This is the next genuine hard problem.
- **Human preference alignment** — add a DPO/RLHF layer on top of GRPO for tone and professionalism.
- **Production simulation** — 200 emails/day, SLA latency budget, concurrent users, priority inbox management. The environment is already multi-session capable (max 20 concurrent, FIFO eviction).
- **Cross-domain transfer** — does an email-triage-trained model transfer better to ticket triage, customer support routing, or code review queues?

---

## Try It Right Now

**One command:**
```bash
curl -X POST https://hk4crprasad-email-triage-env.hf.space/reset \
  -H 'Content-Type: application/json' \
  -d '{"task_id": "hard", "seed": 42}' | python -m json.tool
```

**See the phishing email get caught:**  
Open [/demo](https://hk4crprasad-email-triage-env.hf.space/demo) → tab **🆚 Baseline vs Trained adapter** → Task=hard, Seed=42, Email index=2 → **▶ Run side-by-side**

Watch the zero-shot model flag it as urgent. Watch the trained model call it spam.

---

## Links

| Resource | Link |
|----------|------|
| 🤗 HF Space | [spaces/Hk4crprasad/email-triage-env](https://huggingface.co/spaces/Hk4crprasad/email-triage-env) |
| 🎮 Gradio demo | [/demo](https://hk4crprasad-email-triage-env.hf.space/demo) |
| 🧠 Trained adapter | [Hk4crprasad/email-triage-grpo](https://huggingface.co/Hk4crprasad/email-triage-grpo) |
| 💻 GitHub | [hk4crprasad/my-env](https://github.com/hk4crprasad/my-env) |
| 🎓 Training Colab | [train_grpo.ipynb](https://colab.research.google.com/github/hk4crprasad/my-env/blob/main/notebooks/train_grpo.ipynb) |
| 🧪 Demo & Test Colab | [demo_and_test.ipynb](https://colab.research.google.com/github/hk4crprasad/my-env/blob/main/notebooks/demo_and_test.ipynb) |
| 📊 Reward rubric (live) | `GET /rubric` |
| 🧭 Curriculum (live) | `GET /curriculum` |

---

*Built by Team Ctrl-Alt-Defeat — Haraprasad Hota, Subhendu Samal, Ashutosh Panigrahi — for the Scaler × Meta PyTorch OpenEnv Hackathon 2026, Bangalore.*

*Questions, re-runs, or want to train your own model against our environment? Open an issue on GitHub or ping us on the hackathon Discord.*
