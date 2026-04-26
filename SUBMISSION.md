# Submission — Quick Reference

> Email Triage RL Environment · OpenEnv Hackathon 2026 · Team Ctrl-Alt-Defeat
> Submission deadline: **April 26, 2026, lunch (Bangalore venue)**

---

## What you submit (Google Form, Apr 26)

A Google Form is shared on Apr 26 by Scaler. The form asks for **four URLs** — paste these exactly:

| Field | URL |
|-------|-----|
| Hugging Face Space URL | `https://huggingface.co/spaces/Hk4crprasad/email-triage-env` |
| Colab Notebook link | `https://colab.research.google.com/github/hk4crprasad/my-env/blob/main/notebooks/train_grpo.ipynb` |
| Code repository link | `https://github.com/hk4crprasad/my-env` |
| YouTube video URL **or** HF blog post URL | `https://huggingface.co/blog/Hk4crprasad/email-triage-grpo-blog` |

**Hard rule from the brief**: every URL above **must also be linked from the README.md** of the repo. Judges may grade by reading the README first.

> Only the team lead can submit. Lead = Haraprasad Hota (`haraprasadhota1@gmail.com`).

---

## How judges actually grade you

The brief (`read/3.txt`) defines four scored dimensions:

| Criterion | Weight | What we score on |
|-----------|-------:|------------------|
| Environment Innovation | 40% | novel/creative env that meaningfully tests agent behaviour |
| Storytelling & Presentation | 30% | clear problem framing, engaging demo, easy to follow |
| Showing Improvement in Rewards | 20% | observable evidence of training: curves, before/after, baseline comparison |
| Reward & Training Pipeline | 10% | coherent reward logic + pipeline produces meaningful improvement |

Plus four **mandatory minimums** — missing any is a "serious disadvantage":

- ✅ OpenEnv (latest release) — `openenv-core>=0.2.0` in `requirements.txt`
- ✅ Working training script (TRL or Unsloth, ideally Colab) — `notebooks/train_grpo.ipynb`
- ✅ Evidence of real training — `plots/training_curve.png`, `plots/score_comparison.png`, baseline-vs-trained inside the notebook
- ✅ Mini-blog ≤ 2 min OR YouTube ≤ 2 min — `blog_post.md` (also published on HF Blog)
- ✅ HF Space deploy + README that motivates problem, explains env, shows results — done

---

## What judges will physically do

Per `read/3.txt` §"Judging Criteria" + §"What Makes a Submission Stand Out":

1. **Open the README** in 3–5 minutes; want a clear story (problem → environment → results → why it matters), all auxiliary URLs linked from there.
2. **Click the HF Space URL** — must respond. Try the [`/demo`](https://hk4crprasad-email-triage-env.hf.space/demo) Gradio UI.
3. **`POST /reset`, `POST /step`** to verify the OpenEnv API works.
4. **Open the Colab notebook**, scroll, possibly **re-run the training** to verify it actually trains an agent (not just print fake numbers).
5. **Check the LoRA adapter on HF Hub** to verify the model was actually pushed.
6. **Look at the plots** — they want labelled axes, units, baseline-vs-trained comparison on the same axes.

---

## Pre-submission checklist (run today, in order)

```bash
# 1. Validator green
python scripts/validate_env.py
# expect: 26 passed, 0 failed

# 2. Local server smoke test
uvicorn server.app:app --host 0.0.0.0 --port 7860 &
sleep 5
curl -s http://localhost:7860/health
# expect: {"status":"healthy"}
curl -s -X POST http://localhost:7860/reset \
     -H "Content-Type: application/json" \
     -d '{"task_id":"easy","seed":42}' | head -c 200
# expect: JSON with session_id, observation.emails[5]
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:7860/demo
# expect: 200 (Gradio UI)
kill %1

# 3. Inference produces [START]/[STEP]/[END] format
export HF_TOKEN="hf_..."
export MODEL_NAME="openai/gpt-oss-120b"
python inference.py 2>&1 | grep -E "^\[(START|STEP|END)\]" | head -5
# expect: 5 well-formed structured lines

# 4. Plots present
ls -la plots/
# expect: training_curve.png, score_comparison.png, dimension_breakdown.png, reward_spread.png
```

---

## URLs to test before submitting

Open each, confirm 200 OK and content makes sense:

```
https://huggingface.co/spaces/Hk4crprasad/email-triage-env
https://hk4crprasad-email-triage-env.hf.space/health
https://hk4crprasad-email-triage-env.hf.space/rubric
https://hk4crprasad-email-triage-env.hf.space/demo
https://huggingface.co/Hk4crprasad/email-triage-grpo
https://github.com/hk4crprasad/my-env
https://huggingface.co/blog/Hk4crprasad/email-triage-grpo-blog
https://colab.research.google.com/github/hk4crprasad/my-env/blob/main/notebooks/train_grpo.ipynb
```

---

## Things they explicitly said NOT to do

- Don't include big video files in the env submission on HF Hub — keep size small. Use URL references.
- Don't reinvent OpenEnv. Build *on top of* the framework.
- Don't use reserved tool names (`reset`, `step`, `state`, `close`) for any MCP tools.
- Don't make the README an API doc — make it tell a story.
- Don't push commits after the deadline. Judges pull the env from the URL at submission time. Changes after won't count.

---

## After submission

Per `read/5.txt`:

- 12:00 noon Apr 26 — lunch / submission deadline
- 2:00 PM Apr 26 — mentor round 3 (final)
- 5:00 PM Apr 26 — closing remarks
- 5:30–8:00 PM Apr 26 — open networking / event concludes

Prize: ₹30,000 pool + direct interview opportunity at Meta & Hugging Face AI teams.
