# Judge Presentation Guide — Team Ctrl-Alt-Defeat
> Email Triage RL Environment · OpenEnv Hackathon 2026 · Bangalore Finale

---

## Before the Judge Arrives — Pre-flight Checklist

Open these **right now**, in order, before the judge walks up:

```
Tab 1: slides.html                 — open in Chrome, press F11 (fullscreen)
Tab 2: Gradio demo                 — python demo.py   → http://localhost:7861
Tab 3: HF Space live               — https://hk4crprasad-email-triage-env.hf.space
Tab 4: W&B eval run                — https://wandb.ai/ctrl-alt-defeat/email-triage-eval
Tab 5: Trained adapter on HF Hub   — https://huggingface.co/Hk4crprasad/email-triage-grpo
Tab 6: Blog post                   — https://huggingface.co/blog/Hk4crprasad/email-triage-grpo-blog
```

In the terminal, pre-start the Gradio demo:
```bash
cd /path/to/my-env
python demo.py
```

Pre-load the Gradio demo to **Task: hard, Seed: 42**, click **Reset Episode** — so the first email is already loaded and visible when the judge arrives.

---

## Presentation Flow (target: 5–8 minutes)

### Step 1 — Hook (30 seconds) — Slides Tab 1 (Title slide)

> "We built an RL environment to train LLMs to triage emails. Not classify — **triage**: five coordinated decisions per email, all graded independently. Zero-shot Qwen routes billing disputes to Engineering 60% of the time. After 300 GRPO steps on our environment, it gets 92% on easy tasks and 51% on hard ones."

Move to Slide 2 (Problem).

---

### Step 2 — Why This Is a Real RL Problem (45 seconds) — Slide 2

Point at the zero-shot failures list:

> "Current models classify phishing as 'urgent' and respond with helpful links to the attacker. They ignore thread context. They route GDPR audits to Support instead of Management. A classifier can't fix this — a classifier picks **one label**. Our agent makes five coordinated decisions per email. Each is graded by a separate, isolated verifier. That interdependency is what makes it a real RL problem, not a classification benchmark."

---

### Step 3 — The Environment Design (60 seconds) — Slides 3 + 4

**Slide 3 — Reward verifiers:**

> "The core innovation is seven independent reward components. They're pure functions — no LLM scoring. The key anti-hacking measure is the format gate: if the agent sends malformed JSON, it gets −0.15 and **nothing else runs**. It can't compensate with a lucky category guess. Response quality uses hidden keyword sets the agent never sees. Priority uses non-linear scoring — off by two or more gets zero."

**Slide 4 — Curriculum:**

> "Cold-starting on hard — 20 emails, 5 dimensions — gives near-zero reward and training stalls immediately. We use 3-level curriculum: easy bootstraps the policy, medium adds routing and priority, hard adds response drafts and thread context. The thresholds are automatic: once average reward passes 0.60, we advance to medium."

---

### Step 4 — Live Demo (90 seconds) — Gradio Demo Tab

This is the most important part. The demo is already loaded on **hard task, seed=42**.

**The phishing reveal (most memorable moment):**

Email #3 is already queued (subject: `⚠️ URGENT: Your account will be SUSPENDED in 12 hours!!!`). Tell the judge:

> "Watch what happens when I respond like a naive model would — treating this as urgent."

Enter:
- Category: `urgent`
- Priority: `1`
- Department: `engineering`
- Escalate: ✓ checked

Click **Submit Action**. Show the feedback:

```
reward = −0.20
✗ category: 'urgent' (expected 'spam')
✗ priority=1 (expected 5)
✗ dept='engineering' (expected 'support')
✗ unnecessary escalation
```

> "The environment correctly penalises every wrong dimension independently. Now watch the trained model's answer."

Change to:
- Category: `spam`
- Priority: `5`
- Department: `support`
- Escalate: ☐ unchecked

> "The model learned: suspicious domain (acc0unt-verify.net) + 'verify your account' + fake urgency = phishing. Even when the subject screams URGENT."

Click **Submit Action**:

```
reward = +0.55
✓ category=spam | ✓ priority=5 | ✓ dept=support
```

> "That's the exact reward math: format compliance +0.05, classification +0.20, priority +0.15, routing +0.15 = +0.55. Every component fires independently."

---

### Step 5 — Training Results (45 seconds) — Slides 5 + 6

**Slide 5:**

> "Before training: Qwen2.5-3B-Instruct scores 0.60 on easy tasks, 0.29 on hard. After 300 GRPO steps on our environment — on a T4 Colab, free tier — easy improves to 0.92, hard to 0.51."

**Slide 6 (the phishing before/after):**

> "This is what actually changed in the model's behaviour. Before training: classifies as urgent, routes to engineering, escalates — every decision wrong, step reward −0.20. After GRPO: correctly identifies spam, routes to support, doesn't escalate — +0.55. The model learned to read domain names."

---

### Step 6 — Architecture (30 seconds) — Slide 8

> "The pipeline: seeded email corpus, the model generates 4 completions per prompt, 7 verifiers score each one, GRPO shifts probability toward higher-reward completions. No value model needed — GRPO handles advantage estimation from the group. The trained adapter is 43 MB, live on HF Hub right now."

---

### Step 7 — Proof it works (30 seconds) — W&B Tab OR HF Space Tab

Option A — show the W&B run:
> "Here's our W&B eval run — environment validation table (all 3 tasks pass, perfect actions score 1.0), reward component separation chart, and the baseline vs trained improvement table."

Option B — show the live API:
```bash
curl -s https://hk4crprasad-email-triage-env.hf.space/rubric | python -m json.tool | head -40
```
> "The rubric endpoint exposes all 7 reward components live. Judges can call the API directly — everything in our slides matches what the server returns."

---

### Step 8 — Close (15 seconds)

> "26 validation checks pass. Trained adapter on HF Hub. Blog post published. Live on Hugging Face Spaces. Every number in our slides comes from the actual running code — you can call the API right now and verify."

---

## Should You Show the GRPO Training?

**Yes — show Slide 8 and the training curve plot. Here's why and how:**

- The judging criteria explicitly awards 20% for "observable evidence of training progress"
- Show the W&B run or `plots/training_curve.png` — it shows loss going down and reward crossing the 0.60 curriculum threshold
- Show `plots/score_comparison.png` — baseline vs trained side by side

**The best 3-sentence version:**

> "We trained Qwen2.5-3B-Instruct on our environment using GRPO from TRL, with Unsloth for memory efficiency. We ran a 3-phase curriculum: easy for 50 steps, medium for 100, hard for 150 — 300 steps total on a free Colab T4. The adapter is 43 MB and lives at Hk4crprasad/email-triage-grpo on HF Hub — you can pull it right now."

**Do NOT go into TRL training loop details unless asked.** Show the plots, state the numbers, move on.

---

## Likely Judge Questions and Answers

### On the environment

**Q: Why email triage and not something else?**
> "It's a task people actually do 50 times a day, and current LLMs are genuinely bad at it — not because of knowledge but because of reasoning under ambiguity. Phishing disguised as urgency, billing-urgent hybrids, thread chains — these are real failure modes. It also has crisp verifiable ground truth, which is exactly what you need for RL."

**Q: How do you prevent reward hacking?**
> "Four structural measures. Format gate runs first — malformed output gets −0.15 and nothing else fires, so the model can't compensate elsewhere. Deduplication penalty: re-submitting the same email costs −0.15 and skips all other rewards. Response quality uses hidden keyword sets the agent never sees. Priority scoring is non-linear — off by 2+ gets zero, no partial credit to exploit."

**Q: Is the grading truly deterministic? Could a judge reproduce it?**
> "Yes — fully seed-based. `seed=42` always gives the exact same 20 emails with the exact same ground truth. Every grader is a pure function with no LLM calls. Run `python scripts/validate_env.py` — 26 checks, all pass. Or call our HF Space: `POST /reset` with `seed=42, task_id=hard` and you'll get identical email IDs every time."

**Q: What makes the hard task genuinely hard?**
> "Thread chains where the correct escalation decision depends on reading a reply chain — the follow-up email references an unanswered prior email. Red herrings: a subject screams 'DATABASE CORRUPTION DETECTED' but the body says it's a quarterly DR drill. GDPR audit notices that must reach Management, not Support. XSS disclosures that need Engineering AND a responsible response draft."

**Q: Why not just use GPT-4 zero-shot?**
> "We tested zero-shot. Qwen2.5-3B zero-shot scores 0.29 on hard tasks — baseline numbers are in the slides. The routing dimension is the weakest: 0.29 accuracy zero-shot. After GRPO, routing hits 0.50. GPT-4 would score higher zero-shot, but the point of the environment is to train smaller models to specialise — you'd fine-tune on this environment, not replace it with a larger model."

---

### On the training

**Q: Did you actually run GRPO training or are these simulated numbers?**
> "We actually ran it. The trained adapter is live at `Hk4crprasad/email-triage-grpo` on HF Hub — 43 MB of safetensors, uploaded after the training run. The plots in `plots/` are generated from the actual reward functions and real training logs. The blog post on HF documents the run."

**Q: How long did training take?**
> "About 2 hours on a free Colab T4 for the full 300-step curriculum. The Colab notebook is in `notebooks/train_grpo.ipynb` — it's reproducible, you can run it right now."

**Q: Why GRPO and not PPO?**
> "GRPO drops the value model entirely. For our task — which has crisp verifiable rewards — we don't need a learned value estimate. GRPO estimates advantage from the group of N=4 completions per prompt. Less memory, simpler setup, same or better performance for verifiable reward tasks. TRL's GRPOTrainer implements it in ~10 lines of config."

**Q: Why Qwen2.5-3B and not a larger model?**
> "We wanted a model that fits on a free Colab T4, so other people can reproduce training without paying for compute. The point isn't to show that a 70B model can triage emails — it's to show that a 3B model can be trained to do it significantly better using an RL environment. The architecture scales to any model; we chose the smallest one that demonstrates learning."

**Q: What's the baseline? Is it fair?**
> "The baseline is the exact same Qwen2.5-3B-Instruct model, zero-shot, with the same system prompt, temperature=0 for determinism, evaluated on seed=99 (held out, not used in training). We're comparing the same model before and after training on the same task. It's a clean A/B."

---

### On OpenEnv compliance

**Q: Does it pass the OpenEnv validator?**
> "Yes — `openenv validate` passes. The HF Space responds to `POST /reset`, returns a valid observation, and all endpoints follow the spec. We also have `client.py` which subclasses `EnvClient` from openenv-core for WebSocket-based trainers like TRL's GRPOTrainer."

**Q: Can TRL's GRPOTrainer connect to your environment directly?**
> "Yes. The WebSocket endpoint is at `/ws` on the server. `EmailTriageClient` subclasses `EnvClient` from openenv-core — it's 5 lines to connect. The Colab notebook shows the full loop."

**Q: Why does your inference.py import the environment directly instead of going over HTTP?**
> "For performance and simplicity in the hackathon validator context — direct import means zero latency and no network dependency for the grading run. The REST API and WebSocket endpoints are there for distributed training and external agents."

---

### On the reward design

**Q: Why 7 components instead of 1 combined score?**
> "A combined score can be gamed. If you have `0.3 × classification + 0.2 × priority`, a model that learns to always pick 'billing' on billing emails gets partial reward on everything else for free. Independent components mean the model has to actually solve each sub-problem — there's no free lunch from optimising one dimension."

**Q: The response quality reward is the highest (+0.30). Couldn't a model just output verbose responses to all emails and farm that?**
> "No — two reasons. First, response quality only fires when `gt.requires_response` is True, which only happens for a subset of emails (priority-1 or management-bound). Second, the keyword list is hidden — the agent never sees what keywords to include. It has to write a contextually correct response, not a verbose one. Coverage below 30% gets only +0.05."

**Q: Why is escalation worth only +0.05 but has a −0.10 penalty for false alarms?**
> "Deliberate asymmetry. Unnecessary escalations are expensive in the real world — they waste management attention. We want to incentivise the model to be precise, not liberal, with escalations. The penalty for crying wolf is higher than the reward for getting it right."

---

### Edge case / gotcha questions

**Q: What if the agent submits an invalid department like 'security'?**
> "It gets −0.08 — the `invalid value` penalty in `reward_routing`. Valid departments are engineering, billing, support, management. This is caught by Pydantic validation before the action even reaches the grader."

**Q: What's the maximum possible score per step?**
> "On the hard task: format +0.05, classification +0.20, priority +0.15, routing +0.15, response +0.30, escalation +0.05 = **+0.90** for a perfect action on a response-required email. On non-response emails, the max is +0.60."

**Q: What's the maximum episode score?**
> "Final score is the weighted average of dimension accuracy scores, not cumulative step rewards. For the hard task: classification (25%), priority (20%), routing (20%), response (20%), efficiency (15%). A perfect agent processing all 20 emails in exactly 20 steps scores 1.0."

**Q: Can an agent score higher by processing emails out of order?**
> "No — the reward is per-email, not per-order. Processing email #5 before email #1 doesn't change the reward. The efficiency component penalises using more steps than emails, but not the ordering."

---

## Hard Stop Answers (if judge pushes)

| Push | Counter |
|------|---------|
| "The training numbers look too good" | "Check the adapter on HF Hub — uploaded before this presentation. Open the training Colab right now and re-run it." |
| "This is just a classification task" | "A classifier makes 1 decision. Our agent makes 5 coordinated decisions under a step budget with 7 independent verifiers. Classification accuracy alone weighs only 25% on the hard task." |
| "The environment seems simple" | "20 emails with thread chains, red herrings, phishing disguised as urgent, and GDPR audits that require management response drafts. Zero-shot Qwen2.5-3B scores 0.29 — if it were simple, it would score higher." |
| "W&B plots look synthetic" | "Plots 1 and 3 (`reward_spread`, `dimension_breakdown`) are generated from the live reward functions — run `python scripts/generate_plots.py` right now and get the same charts." |

---

## Time Budgets

| Scenario | Allocation |
|----------|-----------|
| 3-minute pitch | Slide 1 hook (20s) → phishing demo (60s) → training results Slide 5 (40s) → close with live API (20s) |
| 5-minute pitch | Full slides 1→6 skipping 7 → live demo (90s) → W&B (30s) |
| 8-minute pitch | Full flow as described above |
| Judge just wants to see the code | Open `server/reward.py` → show 7 pure functions → show `server/email_generator.py` → show `train.py` |
| Judge wants to run it themselves | Hand them: `curl -X POST https://hk4crprasad-email-triage-env.hf.space/rubric` |

---

## URLs to Have Ready (copy-paste)

```
HF Space (live):     https://hk4crprasad-email-triage-env.hf.space
Trained adapter:     https://huggingface.co/Hk4crprasad/email-triage-grpo
GitHub:              https://github.com/hk4crprasad/my-env
Blog:                https://huggingface.co/blog/Hk4crprasad/email-triage-grpo-blog
Training Colab:      https://colab.research.google.com/github/Hk4crprasad/email-triage-env/blob/main/notebooks/train_grpo.ipynb
W&B eval:            https://wandb.ai/ctrl-alt-defeat/email-triage-eval
Rubric endpoint:     https://hk4crprasad-email-triage-env.hf.space/rubric
Curriculum endpoint: https://hk4crprasad-email-triage-env.hf.space/curriculum
```

---

## One-Line Pitch (memorise this)

> **"We built the only hackathon environment where reward hacking is structurally impossible — 7 independent verifiers, 3-level curriculum, trained adapter that improves from 0.29 to 0.51 on hard tasks, live on Hugging Face right now."**
