---
title: "Multi-Agent Email Triage with GRPO: Cooperation, Theory-of-Mind, and Verifiable RL Rewards"
thumbnail: https://huggingface.co/spaces/Hk4crprasad/email-triage-env/resolve/main/plots/score_comparison.png
authors:
  - user: Hk4crprasad
  - user: subhendusamal
  - user: ashutoshpanigrahi
tags:
  - reinforcement-learning
  - grpo
  - rlvr
  - openenv
  - email-triage
  - curriculum-learning
  - reward-shaping
  - policy-gradient
  - trl
  - unsloth
  - qwen2.5
  - world-modeling
---

# Training an LLM Email Triage Agent with GRPO: Reinforcement Learning from Verifiable Rewards

![Baseline vs GRPO — Qwen3.5-2B score comparison across easy / medium / hard tasks](https://huggingface.co/spaces/Hk4crprasad/email-triage-env/resolve/main/plots/score_comparison.png)

> **📧 Live environment**: [huggingface.co/spaces/Hk4crprasad/email-triage-env](https://huggingface.co/spaces/Hk4crprasad/email-triage-env)  
> **🎮 Interactive demo**: [`/demo`](https://hk4crprasad-email-triage-env.hf.space/demo) — see baseline vs trained side-by-side in your browser  
> **🧠 Trained adapter**: [Hk4crprasad/email-triage-grpo](https://huggingface.co/Hk4crprasad/email-triage-grpo)  
> **💻 Code**: [github.com/hk4crprasad/my-env](https://github.com/hk4crprasad/my-env)  
> **🏆 OpenEnv Hackathon 2026** — Scaler × Meta PyTorch, Grand Finale, Bangalore

---

## TL;DR

We built an **OpenEnv-compliant Multi-Agent RL environment** combining **Theme #1 (Multi-Agent Interactions)** and **Theme #3.2 (Personalized Tasks)**. Three specialist agents — Analyst, Router, Responder — cooperate on each email step using a sequential chain with **theory-of-mind** and **coalition** scoring. We trained `Qwen/Qwen3.5-2B` using **GRPO** on a 3-level curriculum. The result: hard-task triage score improves from **0.29 → 0.59** (+0.30). Individually-correct but inconsistent agent decisions are penalised, creating genuine multi-agent coordination pressure. No value model. No learned reward model. Pure verifiable RL.

---

## 1. Why Multi-Agent RL for Email Triage

### Theme alignment: #1 + #3.2

Our environment spans two themes deliberately:

- **Theme #3.2 (Personalized Tasks)**: Email triage is a canonical personal assistant task — classify, route, escalate, draft responses. It's real, underexplored for RL, and has crisp verifiable rewards.
- **Theme #1 (Multi-Agent Interactions)**: Real triage teams don't use one person who makes all decisions in isolation. They have specialists. An analyst reads the email. A router decides where it goes. A responder drafts the reply. Adding multi-agent cooperation makes the environment dramatically more challenging and more realistic.

The combination is what makes this submission novel: multi-agent coordination on a **real-world task** with **verifiable rewards** — not a game, not a grid world, but a workflow that exists in every organisation.

### Why three agents?

A single agent makes 5 decisions per email (category, priority, department, escalation, response) and gets rewarded on each independently. This already teaches coordination — but it's all internal to one model. Adding separate agents creates:

- **Explicit theory-of-mind pressure**: The Router must model what the Analyst decided and use it. If the Analyst flagged "suspicious domain" but the Router ignores this and routes to engineering, both the routing reward AND the theory-of-mind bonus are lost.
- **Coalition enforcement**: The Responder's draft must be coherent with both the Analyst's category AND the Router's department. A response that acknowledges a "billing dispute" when the Router sent it to engineering gets no coalition bonus.
- **Emergent consistency**: The model learns that coordinated team decisions score higher than individually-correct-but-inconsistent ones.

---

## 1b. Why Reinforcement Learning — Not Fine-Tuning

Before the environment, the question: *why RL at all?*

Email triage looks like a multi-label classification problem. Train on (email, label) pairs, predict labels. This is what almost everyone does with "LLM-based email management" systems. And it fundamentally fails at the hard cases for a structural reason.

**The structural problem with supervised fine-tuning (SFT) for triage:**

SFT teaches the model to imitate a demonstrator. The demonstrator labels `category=billing`. The model learns to output `billing` given similar emails. What it does **not** learn:

1. **Coordination between decisions.** Category, priority, routing, escalation, and response are five correlated decisions. SFT trains each to match a label independently. But routing `billing` disputes to `engineering` is a specific mistake that only appears in the *joint* error — category might be right while routing is wrong. SFT on separate labels can't teach the joint policy.

2. **Long-horizon credit assignment.** In a triage session, processing 5 easy emails before getting to the critical one matters. SFT on individual emails has no concept of "I should have saved steps for the hard email at the end."

3. **Counterfactual reasoning.** The model should learn: "if I had said `urgent` here instead of `billing`, I would have gotten routing credit back from engineering." SFT provides no counterfactuals.

4. **Genuine ambiguity.** Some emails don't have a single right answer. They have a space of defensible actions with different reward profiles. RL explores that space; SFT collapses it to one demonstration.

**Why RL is the right tool:**

RL directly optimises the *policy* — the mapping from state (inbox + email + context) to action (triage decision). The reward signal is the ground truth we care about. GRPO updates the policy towards higher-reward completions across a group of rollouts, without needing a value function or a learned reward model.

This is the same insight behind DeepSeek-R1, Qwen-RLVR, and the broader RLVR (RL from Verifiable Rewards) paradigm: when you have **crisp, deterministic verifiers**, RL is strictly better than SFT because it can explore the full policy space, not just imitate a fixed demonstration.

---

## 2. The MDP Formulation

We model email triage as a finite-horizon Markov Decision Process (MDP):

### State Space  S

At each timestep *t*, the agent observes:

```
S_t = {
  emails_remaining : List[Email],    # unprocessed emails in inbox
  inbox_stats      : {total, processed, unprocessed},
  task_description : str,            # which task (easy/medium/hard) + instructions
  action_feedback  : str,            # feedback on S_{t-1} action (✓/✗/~)
  step_reward      : float,          # reward from S_{t-1} action
  cumulative_reward: float,          # Σ r_1..r_{t-1}
  steps_remaining  : int,            # horizon H - t
  done             : bool
}
```

The state is **partially observable**: the agent sees email content but not the ground-truth labels or the hidden keyword sets used for response quality grading. This is intentional — it forces the agent to learn to *infer* the right action from the email content, not look up the answer.

### Action Space  A

At each step, the agent selects a single action — a triage decision for one email:

```
A_t = {
  email_id      : str,     # which email to process (from emails_remaining)
  category      : str,     # spam | billing | technical | general | urgent
  priority      : int,     # 1 (critical) → 5 (lowest)
  department    : str,     # engineering | billing | support | management
  response_draft: str|null, # free-text reply (required on hard task for urgent emails)
  escalate      : bool     # notify management immediately?
}
```

The action space is a **structured combinatorial product** — `5 categories × 5 priorities × 4 departments × 2 escalation = 200+ discrete combinations`, plus the continuous response draft. This is not tractable for tabular RL; it requires a parameterised policy (an LLM).

### Reward Function  R(s, a)

The reward is a **sum of 8 independent verifiable components**. See Section 4 for the full design.

### Transition Dynamics  T(s' | s, a)

The environment is **deterministic given a seed**: `T(s_{t+1} | s_t, a_t)` removes the processed email from the inbox, updates stats, and returns the next state. No stochasticity in transitions — all randomness is in the agent's policy.

This determinism is crucial for reproducibility: `seed=42` always produces the same inbox, enabling fair comparison across policies.

### Horizon  H

| Task | Emails | Max steps H | Optimal steps |
|------|-------:|------------:|:-------------:|
| `easy` | 5 | 10 | 5 |
| `medium` | 10 | 25 | 10 |
| `hard` | 20 | 40 | 20 |

An efficiency component rewards completing all emails in exactly H_optimal steps. Wasted steps reduce the inbox_completion bonus.

### Objective

The RL objective is standard expected return maximisation:

```
π* = argmax_π  E_{τ ~ π} [ Σ_{t=0}^{H} γ^t · R(s_t, a_t) ]
```

We use γ=1 (undiscounted) since the horizon is short and all decisions within an episode have equal importance.

---

## 3. The Policy: LLM as a Structured Action Sampler

The policy π is an LLM: `π(a | s) = P_θ(JSON_action | prompt(s))`.

The LLM receives the state as a structured prompt:

```
[SYSTEM]
You are an expert email triage agent. For each email, respond with a single 
valid JSON object:
{
  "email_id": "<id>",
  "category": "<spam|billing|technical|general|urgent>",
  "priority": <1-5>,
  "department": "<engineering|billing|support|management>",
  "response_draft": "<reply text or null>",
  "escalate": <true|false>
}
[Semantic guidance for each field...]

[USER]
TASK: [task description]
Email ID: hard_003
From: security-alerts@support-team-verification.net
Subject: ⚠️ URGENT: Your account will be SUSPENDED in 12 hours!!!
Body: Your account has been flagged...click bit.ly/verify-now...
```

The LLM samples a completion. That completion is parsed as a JSON action and evaluated by the reward function. The gradient flows back through the token probabilities of the sampled completion.

**Why LLM-as-policy is the right architecture for this task:**

- The action space is combinatorial and depends on reading comprehension — a tabular policy can't read emails
- The policy must generalise across novel emails (not seen during training) — an LLM's pre-trained world knowledge is the prior
- The structured JSON output format enforces action validity without a separate parser
- LoRA fine-tuning allows us to adapt the policy efficiently without catastrophic forgetting

---

## 4. Reward Engineering: The Core RL Challenge

Reward design is where most RL projects succeed or fail. Here's exactly how we designed ours.

### 4.1 The credit assignment problem

Triage has 5 sub-tasks per email: classify, prioritise, route, respond, escalate. A naive combined reward like:

```python
reward = 0.3*classify_correct + 0.2*priority_correct + 0.2*route_correct + ...
```

creates **reward aliasing** — the policy can get positive reward by being good at one dimension while ignoring others. Specifically:

- **Spam attack**: Always predict `category=billing` on billing emails → positive classification reward bleeds into every timestep, masking poor routing
- **Escalation farming**: Always set `escalate=True` → gets TP reward (+0.10) on the 20% of emails that need escalation, net positive even with FP penalties
- **Format exploit**: Generate a well-formed JSON with a random `email_id` → occasionally lands on a valid ID and gets classification credit

### 4.2 Independence as the solution

The solution is to make each component **informationally independent**: no component's reward can be improved by optimising a different component.

```python
# server/reward.py — all 8 components are pure Python, no LLM

def reward_format_compliance(action, valid_email_ids):
    """Structural gate. Wrong email_id → immediate return, nothing else fires."""
    if action['email_id'] not in valid_email_ids:
        return -0.10  # can't get ANY other reward with hallucinated email_id

def reward_classification(action, gt):
    """Semantic distance on category space."""
    dist = CATEGORY_SEMANTIC_DISTANCE[gt.category][action['category']]
    # Non-binary: rewards semantic proximity, not just exact match
    return {0: +0.30, 1: +0.10, 2: 0.00}.get(dist, -0.08)

def reward_escalation(action, gt):
    """F1-style: both FP and FN penalised separately."""
    should_escalate = gt.department == "management" or gt.priority == 1
    agent_escalates = bool(action.get("escalate", False))
    if agent_escalates and should_escalate:     return +0.10  # TP
    if not agent_escalates and should_escalate: return -0.05  # FN: missed critical
    if agent_escalates and not should_escalate: return -0.10  # FP: unnecessary noise
    return +0.03                                              # TN: correct restraint

def reward_response_quality(action, gt):
    """Hidden keyword coverage. Agent can't reverse-engineer the keywords."""
    keywords = gt.expected_keywords   # NOT in the agent's prompt
    draft    = (action.get("response_draft") or "").lower()
    coverage = sum(kw in draft for kw in keywords) / max(len(keywords), 1)
    if coverage >= 0.70: return +0.35
    if coverage >= 0.50: return +0.25
    if coverage >= 0.30: return +0.15
    return +0.05
```

The complete reward table, with the theoretical reasoning for each component's min/max:

| Component | Max | Min | RL Design Rationale |
|-----------|----:|----:|---------------------|
| `format_compliance` | +0.05 | −0.15 | **Gate**: runs first, shorts all others if invalid. Makes format a prerequisite, not a bonus |
| `anti_reprocessing` | 0.00 | −0.15 | **Step-cost**: agent can't exploit a single known-good email. Creates pressure to process all emails |
| `classification` | +0.30 | −0.15 | **Semantic gradient**: distance-based reward creates dense signal, not binary correct/wrong |
| `priority` | +0.20 | −0.10 | **Non-linear**: off-by-2+ gets 0 (not partial). Removes gradient toward median-guessing |
| `routing` | +0.20 | −0.15 | **Semantic gradient**: department distance rewards proximity — near-misses get 0, not −0.15 |
| `response_quality` | +0.35 | −0.15 | **Hidden target**: keywords not shown in prompt, preventing reverse-engineering |
| `escalation` | +0.10 | −0.10 | **Asymmetric F1**: FP costs more (−0.10) than FN (−0.05) — mirrors real cost of over-escalation |
| `inbox_completion` | +0.05 | 0.00 | **Episode bonus**: reward for processing ALL emails, creating efficiency pressure |

### 4.3 Reward shaping for dense signals

Sparse rewards (0/1 at episode end) are the hardest RL problem. Our rewards are **dense**: every step returns signal on every dimension the agent attempted. This is possible because the reward functions are deterministic Python — they don't need a model to evaluate.

The verifier separation chart confirms the design works empirically:

![Reward Spread](https://huggingface.co/spaces/Hk4crprasad/email-triage-env/resolve/main/plots/reward_spread.png)

*Perfect actions (green) score near-maximum on every verifier. Random actions (red) hover near zero. Adversarial actions (always-wrong-valid answer) get consistently penalised across all components. Independence is what makes this separation possible.*

---

## 5. The RL Algorithm: GRPO

### 5.1 Why GRPO over PPO

PPO (Proximal Policy Optimisation) requires a **value network** V(s) to estimate expected return for advantage computation. For email triage:

- V(s) would need to model "given this inbox state, what's the expected total reward?"
- Training V(s) requires separate rollouts, memory, and a value loss
- For short-horizon tasks with crisp verifiable rewards, V(s) adds complexity without benefit

**GRPO** (Group Relative Policy Optimisation) drops V(s) entirely. It estimates advantage from **group contrast**: generate N completions per prompt, use their reward distribution to compute relative advantage.

```
For prompt x and N completions {o_1, ..., o_N}:

advantage_i = (r_i - mean({r_1,...,r_N})) / std({r_1,...,r_N})

policy_gradient ∝ Σ_i [ advantage_i · ∇_θ log π_θ(o_i | x) ]
```

This is exactly the DeepSeek-R1 / Qwen-RLVR setup for math reasoning — and it works for any task with deterministic verifiers.

### 5.2 The GRPO update step

```python
# Simplified GRPO update (from TRL GRPOTrainer)
for batch in dataloader:
    prompt = batch['prompt']                    # email + task instructions
    
    # Generate N rollouts from current policy
    completions = model.generate(prompt, n=N, temperature=T)
    
    # Score each completion with ALL 8 reward functions (same as production server)
    rewards = [sum(fn([c], **batch) for fn in REWARD_FUNCTIONS) for c in completions]
    
    # Group-relative advantage (no value network needed)
    mean_r, std_r = mean(rewards), std(rewards)
    advantages    = [(r - mean_r) / (std_r + 1e-8) for r in rewards]
    
    # Policy gradient with clipped importance weights (PPO-style stability)
    ratio  = π_θ(completion) / π_θ_old(completion)      # importance weight
    loss   = -min(ratio * adv, clip(ratio, 1-ε, 1+ε) * adv)
    
    # Update policy
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=0.1)    # tight clipping
    optimizer.step()
```

The key properties:
- **No value model** — advantage is computed from N=4 group rollouts
- **Importance-weighted clipping** — same PPO-clip trick for training stability  
- **Tight grad clipping** (`max_norm=0.1`) — prevents large updates from lucky high-reward completions
- **Same reward functions as production server** — single source of truth

### 5.3 Why N=4 is enough

We tried N=8 expecting faster learning from higher variance estimation. In practice N=4 achieved the same final scores with 2× faster rollout generation.

The reason: with 8 independent reward components, even 4 completions produce enough reward variance for GRPO to find a reliable gradient direction. The reward isn't binary — a completion can score anywhere from strongly negative to strongly positive across all 8 components simultaneously.

### 5.4 Temperature strategy

`T=1.0` throughout training. This is higher than inference temperature (`T=0` for greedy decoding) and intentional: GRPO requires **diversity** in the group. If all N completions are nearly identical (which happens at low T), the advantages are near zero and no gradient flows.

The diversity-learning tradeoff in GRPO:
```
Low T → similar completions → low advantage variance → weak gradient → slow learning
High T → diverse completions → high advantage variance → strong gradient but noisy
T=1.0 → empirically optimal for verifiable-reward tasks
```

---

## 6. Curriculum RL: Solving the Cold-Start Problem

### 6.1 The cold-start failure mode

Our first training attempt: cold-start on `hard` (20 emails, 5 dimensions). After 100 steps, reward was stuck near zero.

The RL diagnosis: the **policy never gets sufficient positive signal to reinforce**. GRPO's group advantage is zero when all N completions score equally badly. No gradient, no learning.

This is the cold-start problem in RL: when the initial policy is too far from any rewarding region of the action space, gradient-based methods fail. For tabular RL the solution is exploration bonuses; for LLM RL the solution is curriculum.

### 6.2 Curriculum as reward shaping

Curriculum learning is a form of **potential-based reward shaping**. The environment is modified (easier task → smaller action space → higher reward probability), and the policy is gradually transferred to harder versions.

Our curriculum:

```
easy (5 emails, classify only)
    ↓  advance when avg_reward > 0.60
medium (10 emails, + priority + routing)
    ↓  advance when avg_reward > 0.50
hard (20 emails, all 5 dimensions + thread context + red herrings)
```

**Why 0.60 and 0.50?**  
These thresholds ensure the policy has genuinely learned the current task before adding complexity. At 0.60 on `easy`, the model is correctly classifying ~80% of emails — a solid foundation for learning routing in `medium`.

### 6.3 What each stage teaches

**Stage 1: easy (200 steps)**

The agent learns:
- **Format compliance first.** By step 50, the format gate is clearing >95% of the time. The agent has learned to copy `email_id` exactly and output valid JSON.
- **Basic category semantics.** By step 100, classification accuracy reaches ~75%. Spam detection (the easiest signal — suspicious domains, lottery language) comes first. Technical/billing disambiguation takes longer.
- **Step-cost awareness.** The `anti_reprocessing` penalty teaches the agent not to submit the same email twice — even on a 5-email task, this fires several times early in training.

**Stage 2: medium (100 steps)**

The agent learns:
- **Department routing.** The hardest dimension zero-shot. GRPO forces the model to learn domain-knowledge mappings: XSS → engineering, refund → billing, GDPR → management. These aren't in the standard pre-training distribution for a 3B model.
- **Priority calibration.** The graduated reward (not binary) creates pressure for accurate priority assignment. The model learns not to call everything priority=1 (which gets TP escalation but costs priority reward on non-critical emails).

**Stage 3: hard (100 steps)**

The agent learns:
- **Phishing detection.** The suspicious-domain + URL-shortener + fake-urgency pattern (see Section 7).
- **Red herring suppression.** `[TEST ENV]` in the body means `category=general`, not `urgent` — even when the subject screams "CRITICAL".
- **Response draft quality.** With hidden keywords, the agent can't memorise the expected output. It learns that a useful response covers the topic of the complaint (refund, outage, GDPR deletion) with relevant terminology.
- **Escalation precision.** The F1-style reward discourages over-escalation. The model learns: management/priority=1 → escalate; everything else → don't.

---

## 7. The Phishing Case: RL Learning World Models

The phishing email is the clearest demonstration that RL is learning a **world model**, not pattern-matching.

### The email (hard task, seed=42, index 2)

```
From:    security-alerts@support-team-verification.net
Subject: ⚠️ URGENT: Your account will be SUSPENDED in 12 hours!!!

Dear valued customer,

Your account has been flagged for suspicious activity. To prevent suspension,
you must verify your identity immediately.

Click here to verify: bit.ly/account-verify-secure

Failure to verify within 12 hours will result in permanent account termination.

— Security Team
```

### Before training (zero-shot Qwen3.5-2B)

```json
{
  "email_id": "hard_003",
  "category": "urgent",
  "priority": 1,
  "department": "engineering",
  "escalate": true
}
```
**Reward: −0.28**

The model read "URGENT" + "SUSPENDED" + "12 hours" and pattern-matched to `urgent/priority=1`. It never modelled *who sends this kind of email* or *what `bit.ly` implies*.

### After GRPO training (400 steps)

```json
{
  "email_id": "hard_003",
  "category": "spam",
  "priority": 5,
  "department": "support",
  "escalate": false
}
```
**Reward: +0.55**

The model learned four signals — from reward pressure, not instructions:

| Signal | World knowledge required |
|--------|--------------------------|
| `support-team-verification.net` | Legitimate corporate domains don't include "verification" in TLD. This is a lookalike domain. |
| `bit.ly/` redirect | Real security teams link directly to their domain. URL shorteners in security emails = phishing. |
| "verify your account" | The credential phishing template. Combined with the domain, this confirms spam. |
| Artificial urgency ("12 hours", "permanent termination") | Urgency manufacture is a phishing technique. Real account issues don't come with arbitrary countdowns. |

**Why this proves world modeling:** The model didn't see these four signals in the reward function — the reward function only knows `category == "spam"`. The model had to learn *why spam* from reward pressure across many similar emails. That's world model learning, not label memorisation.

You can observe this in the Gradio demo — `hard` task, seed=42, email index=2, click "▶ Run side-by-side":

![Demo comparison](https://huggingface.co/spaces/Hk4crprasad/email-triage-env/resolve/main/plots/score_comparison.png)

---

## 8. Results: Policy Improvement

### 8.1 Final scores (held-out seed=99)

![Score Comparison](https://huggingface.co/spaces/Hk4crprasad/email-triage-env/resolve/main/plots/score_comparison.png)

| Task | Emails | π_baseline (0-shot) | π_trained (GRPO) | Δ |
|------|-------:|:-------------------:|:----------------:|:---:|
| `easy`   | 5  | 0.60 | **0.80** | **+0.20** |
| `medium` | 10 | 0.38 | **0.61** | **+0.23** |
| `hard`   | 20 | 0.29 | **0.59** | **+0.30** |

*The improvement grows with task difficulty — the hallmark of genuine policy learning, not in-context memorisation.*

### 8.2 Training dynamics

![Training Curve](https://huggingface.co/spaces/Hk4crprasad/email-triage-env/resolve/main/plots/training_curve.png)

Key RL observations from the training curve:

- **Step 0–50**: High variance, format gate firing frequently. Reward near zero.
- **Step 50–150**: Format compliance learned (>95%). Classification begins improving. Reward climbs steadily.
- **Step ~180**: Easy task reward crosses 0.60 threshold. **Curriculum advances to medium.** Reward drops (new, harder task).
- **Step 180–280**: Routing learns quickly — the reward jump from 0.39 to ~0.55 on routing happens in these 100 steps.
- **Step ~280**: Medium reward crosses 0.50. **Curriculum advances to hard.** Reward drops again.
- **Step 280–400**: Phishing detection, red herring suppression, escalation precision learned.

The curve is smooth — no reward hacking spikes — because 8 independent verifiers prevent any single component from being exploited.

### 8.3 Per-dimension improvement (medium task)

![Dimension Breakdown](https://huggingface.co/spaces/Hk4crprasad/email-triage-env/resolve/main/plots/dimension_breakdown.png)

| Dimension | π_baseline | π_trained | Δ | Interpretation |
|-----------|:----------:|:---------:|:---:|----------------|
| Classification | ~0.65 | ~0.82 | +0.17 | GRPO learns phishing patterns + billing/urgent disambiguation |
| **Routing** | ~0.39 | **~0.60** | **+0.21** | Largest gain — domain-knowledge mappings learned from RL |
| Priority | ~0.48 | ~0.66 | +0.18 | Graduated reward teaches calibration, not just exact match |
| Escalation | ~0.55 | ~0.69 | +0.14 | F1 reward discourages systematic over-escalation |

**Routing improves the most.** This is the dimension that requires the most specialised world knowledge — which organisational department handles which issue type. GRPO learned these mappings from reward signal alone in ~100 medium-task steps.

---

## 8b. Multi-Agent Architecture

### The 3-Agent Chain

```
Email
  │
  ▼
┌──────────────────────────────────────────────────┐
│  Agent 1: Analyst  (Qwen3.5-2B, role=analyst)  │
│                                                   │
│  Input:  email content                            │
│  Output: {category, priority, signals[], conf}    │
│                                                   │
│  Trains on: classification + priority accuracy   │
│             + signal extraction bonus             │
└──────────────────┬───────────────────────────────┘
                   │ analyst output passed as context
                   ▼
┌──────────────────────────────────────────────────┐
│  Agent 2: Router   (same model, role=router)      │
│                                                   │
│  Input:  email + Analyst's {category, signals}    │
│  Output: {department, escalate, reason}           │
│                                                   │
│  Trains on: routing accuracy                      │
│             + theory-of-mind: did it use signals? │
│             + consistency with analyst category   │
└──────────────────┬───────────────────────────────┘
                   │ analyst + router context passed
                   ▼
┌──────────────────────────────────────────────────┐
│  Agent 3: Responder (same model, role=responder)  │
│                                                   │
│  Input:  email + Analyst output + Router output   │
│  Output: {response_draft, tone}                   │
│                                                   │
│  Trains on: response quality                      │
│             + coalition: consistent with both     │
└──────────────────────────────────────────────────┘
                   │
                   ▼
          Combined step reward
          = individual rewards
          + coordination bonus
          + theory-of-mind bonus
          + coalition score
```

### Multi-Agent Reward Components

| Reward | Max | What it measures |
|--------|----:|-----------------|
| Individual (×3) | varies | Each agent's accuracy on its specialty |
| `coordination` | +0.10 | Analyst category ↔ Router department semantically aligned |
| `theory_of_mind` | +0.05 | Router reason references Analyst signals |
| `coalition` | +0.08 | Responder draft consistent with both upstream agents |
| Disagreement | −0.08 | Major analyst/router mismatch |

### Role-conditioned GRPO

All three agents are the **same `Qwen/Qwen3.5-2B` model** trained with different system prompts. This is role-conditioned training — the model learns to behave differently based on which role it's playing.

Each email generates 3 training examples (one per role). The dataset is role-labelled so the reward functions can apply role-appropriate scoring:

```python
# From train.py
MULTI_AGENT_REWARD_FUNCTIONS = [
    reward_format,              # JSON validity gate
    reward_coordination_grpo,   # role-aware: analyst→classification, router→routing+ToM, responder→coalition
    reward_response,            # response quality for responder role
]
```

The `reward_coordination_grpo` function checks the agent's `role` field in kwargs and applies the appropriate reward logic. This is the key innovation: a single reward function that scores differently based on which agent is being trained.

### Why Qwen3.5-2B for multi-agent?

`Qwen3.5-2B` in 4-bit NF4 uses ~1.5 GB VRAM and generates a JSON response in ~1.5 seconds. One full 3-agent step takes ~5 seconds — fast enough for real training, and far cheaper than running a 7B model.

The 1B model is also sufficient to show measurable RL improvement. The architecture is what matters, not the parameter count.

---

## 9. RL Training Configuration

Full hyperparameter table with rationale:

```python
GRPOConfig(
    # Curriculum stages: 200 easy + 100 medium + 100 hard
    max_steps                   = 200,           # per stage; 400 total

    # Batch configuration
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,             # effective batch = 4 prompts/update
    num_generations             = 4,             # N rollouts per prompt for group contrast

    # Policy gradient parameters
    learning_rate               = 3e-6,          # conservative — GRPO sensitive to LR
    lr_scheduler_type           = "cosine",      # smooth decay for fine-tuning
    warmup_steps                = 20,            # stabilise baseline before large updates
    max_grad_norm               = 0.1,           # tight — prevents updates dominated by lucky rollouts

    # Sampling (generation policy)
    max_new_tokens              = 320,           # enough for full JSON + response draft
    temperature                 = 1.0,           # high diversity needed for group advantage
    top_p                       = 0.95,          # nucleus sampling for coherent generation

    # Infrastructure
    use_vllm                    = False,         # Unsloth handles fast generation
    save_total_limit            = 2,             # disk management
    seed                        = 42,
)
```

**LoRA configuration (Unsloth):**
```python
FastLanguageModel.get_peft_model(
    model,
    r                = 32,           # rank: expressive enough for multi-dimension learning
    lora_alpha       = 64,           # α = 2r → standard scaling for LoRA
    lora_dropout     = 0.05,         # light regularisation
    target_modules   = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],  # all linear layers
    use_gradient_checkpointing = 'unsloth',  # 40% VRAM reduction
)
```

---

## 10. OpenEnv Integration

The environment implements the full [OpenEnv](https://github.com/meta-pytorch/OpenEnv) spec — the same interface any TRL/Unsloth/ART/Oumi trainer connects to:

```python
# Standard OpenEnv interface — works with any openenv-core trainer
from client import EmailTriageClient
from models import EmailAction

async with EmailTriageClient("https://hk4crprasad-email-triage-env.hf.space") as env:
    # Reset: returns initial observation
    result = await env.reset(task_id="hard", seed=42)
    
    while not result.done:
        # Agent policy: LLM generates action from observation
        action = policy(result.observation)
        
        # Step: environment transitions, returns reward
        result = await env.step(action)
        
        # result.reward = 8-component composite reward
        # result.observation = next state (remaining emails, feedback, stats)
```

The WebSocket protocol means training can run **anywhere** — Colab, local machine, cloud GPU — while the environment runs on the HF Space. This separation is what makes OpenEnv powerful: the environment is a service, the trainer is a client.

---

## 11. Reproducibility

Every result in this post can be independently verified:

| Claim | Verification | Time |
|-------|-------------|------|
| 8 independent reward components | `curl .../rubric` | 5 s |
| Reward verifier separation | `python scripts/generate_plots.py` | 30 s |
| 26-check validation suite | `python scripts/validate_env.py` | 60 s |
| Training results | Run [train_grpo.ipynb](https://colab.research.google.com/github/hk4crprasad/my-env/blob/main/notebooks/train_grpo.ipynb) | 45 min |
| Phishing demo | [/demo](https://hk4crprasad-email-triage-env.hf.space/demo), Tab 2, hard seed=42 idx=2 | 2 min |

The reward functions used during training (`from train import REWARD_FUNCTIONS`) are **byte-identical** to the ones the production server uses to evaluate actions. There is no train/eval reward gap — a single source of truth.

---

## 12. What We Learned from the RL

**Independent verifiers prevent reward hacking.** In early experiments with a combined reward, the model systematically over-escalated: it learned that `escalate=True` on the 20% of emails that genuinely needed escalation was a net positive, even with FP penalties on the other 80%. Independent verifiers remove this — FP escalation always costs −0.10 with no compensation from any other component.

**SFT warm-up hurt.** We tried SFT on 50 expert demonstrations before GRPO. The model became confident in wrong routing decisions (the SFT demonstrations had some routing errors) and resisted GRPO updates. Cold-starting from the instruction-tuned model with curriculum worked better — the `easy` stage serves the same bootstrapping purpose as SFT warm-up.

**The curriculum threshold matters.** We tried 0.50 as the easy→medium threshold. The routing dimension never improved because the model advanced before mastering classification — and classification is the foundation routing builds on. Raising to 0.60 fixed it.

**Temperature scheduling would help.** We used fixed T=1.0. In the late stages (hard task, step 350+), the model was generating mostly-correct actions and the exploration from T=1.0 was adding noise rather than useful variance. A temperature schedule (1.0 → 0.7 over training) would likely smooth the final learning curve.

---

## 13. Future RL Directions

| Direction | RL Framing | Difficulty |
|-----------|-----------|------------|
| Multi-turn thread context | **Partial observability** — agent sees only latest email but must infer thread history | Medium |
| Scale to 7B/8B | Same environment, larger policy — tests RL scaling laws | Low |
| Human preference alignment | DPO/RLHF layer on top of GRPO for tone and professionalism | Medium |
| Production simulation | 200 emails/day, SLA latency budget — **constrained RL** | High |
| Hierarchical policy | Meta-policy decides task difficulty; sub-policy triages | High |

The environment is live and reusable. Any team can train their own model against it:

```bash
# Connect any model to our environment via openenv-core
from openenv.core import GenericEnvClient
env = GenericEnvClient("https://hk4crprasad-email-triage-env.hf.space")
```

---

## Links

| Resource | Link |
|----------|------|
| 🤗 Live environment | [spaces/Hk4crprasad/email-triage-env](https://huggingface.co/spaces/Hk4crprasad/email-triage-env) |
| 🎮 Gradio demo | [/demo](https://hk4crprasad-email-triage-env.hf.space/demo) |
| 🧠 Trained LoRA adapter | [Hk4crprasad/email-triage-grpo](https://huggingface.co/Hk4crprasad/email-triage-grpo) |
| 💻 GitHub | [hk4crprasad/my-env](https://github.com/hk4crprasad/my-env) |
| 📊 Reward rubric (live JSON) | [`GET /rubric`](https://hk4crprasad-email-triage-env.hf.space/rubric) |
| 🧭 Curriculum schedule | [`GET /curriculum`](https://hk4crprasad-email-triage-env.hf.space/curriculum) |
| 🎓 Training Colab | [train_grpo.ipynb](https://colab.research.google.com/github/hk4crprasad/my-env/blob/main/notebooks/train_grpo.ipynb) |
| 🧪 Demo & Test Colab | [demo_and_test.ipynb](https://colab.research.google.com/github/hk4crprasad/my-env/blob/main/notebooks/demo_and_test.ipynb) |

---

*Team Ctrl-Alt-Defeat — Haraprasad Hota, Subhendu Samal, Ashutosh Panigrahi*  
*Scaler × Meta PyTorch OpenEnv Hackathon 2026, Bangalore*
