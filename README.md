# Email Triage Environment 📧

An **OpenEnv-compliant** reinforcement learning environment where AI agents learn to triage emails — classify, prioritise, route, and respond to incoming messages.

Built for the **OpenEnv AI Hackathon** (Meta × Hugging Face × PyTorch × Scaler).

---

## 🎯 Environment Description

**Email triage** is a universal real-world task performed by millions of professionals daily. This environment simulates an inbox where an AI agent must:

1. **Classify** emails into categories (spam, billing, technical, general, urgent)
2. **Prioritise** by severity (1 = critical → 5 = low)
3. **Route** to the correct department (engineering, billing, support, management)
4. **Respond** to critical emails with drafted replies
5. **Escalate** when management attention is needed

The environment provides **meaningful partial rewards** at every step — not just binary end-of-episode scoring.

---

## 🏗️ Architecture

```
email-triage-env/
├── models.py                 # Pydantic Action, Observation, State models
├── server/
│   ├── app.py               # FastAPI server (HTTP endpoints)
│   ├── environment.py       # Core EmailTriageEnvironment
│   ├── email_generator.py   # Deterministic email generation
│   ├── tasks.py             # Task definitions (easy/medium/hard)
│   ├── graders.py           # Deterministic graders (0.0–1.0)
│   └── reward.py            # Step-wise reward shaping
├── inference.py             # Baseline LLM agent
├── openenv.yaml             # OpenEnv manifest
├── Dockerfile               # Container image
├── pyproject.toml           # Package config
├── requirements.txt         # Dependencies
└── README.md                # This file
```

---

## 📋 Action Space

The agent sends an `EmailAction` for each email:

| Field            | Type     | Required | Description                                         |
|------------------|----------|----------|-----------------------------------------------------|
| `email_id`       | `str`    | ✅       | ID of the email to act on                            |
| `category`       | `str`    | ✅       | `spam \| billing \| technical \| general \| urgent`  |
| `priority`       | `int`    | Medium+  | 1 (critical) to 5 (low)                              |
| `department`     | `str`    | Medium+  | `engineering \| billing \| support \| management`    |
| `response_draft` | `str`    | Hard     | Draft reply for critical emails                      |
| `escalate`       | `bool`   | Hard     | Flag for management escalation                       |

---

## 👁️ Observation Space

The agent receives an `EmailObservation`:

| Field               | Type          | Description                              |
|---------------------|---------------|------------------------------------------|
| `emails`            | `list[dict]`  | Unprocessed emails with full content      |
| `inbox_stats`       | `dict`        | Total/processed/unprocessed counts        |
| `task_id`           | `str`         | Current task identifier                   |
| `task_description`  | `str`         | Human-readable task instructions          |
| `action_feedback`   | `str`         | Feedback on last action                   |
| `step_reward`       | `float`       | Reward from last action                   |
| `cumulative_reward` | `float`       | Total reward so far                       |
| `steps_remaining`   | `int`         | Steps before episode truncation           |
| `done`              | `bool`        | Whether episode has ended                 |

Each email contains: `email_id`, `sender`, `sender_name`, `subject`, `body`, `timestamp`, `has_attachment`, `is_reply`, `thread_id`.

---

## 📝 Tasks

| Task | Difficulty | Emails | Max Steps | What to Do |
|------|-----------|--------|-----------|------------|
| **Basic Classification** | Easy | 5 | 10 | Classify each email into a category |
| **Priority & Routing** | Medium | 10 | 25 | Classify + assign priority + route to department |
| **SLA Triage Under Pressure** | Hard | 20 | 40 | Full triage with response drafting, thread awareness, and time pressure |

### Difficulty Progression
- **Easy**: Clear-cut emails with obvious category indicators
- **Medium**: Ambiguous emails, multi-issue complaints, phishing disguised as legitimate
- **Hard**: Thread chains, red herrings, compliance audits, security vulnerabilities, and mandatory response drafting

---

## 🏆 Scoring

### Easy Task
- Classification accuracy (100%)

### Medium Task
- Classification accuracy (40%) + Priority accuracy (30%) + Routing accuracy (30%)

### Hard Task
- Classification (25%) + Priority (20%) + Routing (20%) + Response quality (20%) + Time efficiency (15%)

All scores are deterministic and range from 0.0 to 1.0.

---

## 🚀 Setup & Usage

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Test health check
curl http://localhost:7860/health

# Reset environment (easy task)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "seed": 42}'

# Step with an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"email_id": "abc123", "category": "spam"}}'
```

### Docker

```bash
# Build
docker build -t email-triage-env .

# Run
docker run -p 7860:7860 email-triage-env
```

### Run Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="your-model-name"
export HF_TOKEN="your-hf-token"

python inference.py
```

---

## 📊 Baseline Scores

| Task   | Score  | Notes                         |
|--------|--------|-------------------------------|
| Easy   | ~0.80  | Most emails clearly classified |
| Medium | ~0.55  | Ambiguous cases challenge LLMs  |
| Hard   | ~0.40  | Thread context and response drafting are difficult |

*(Scores may vary by model. Above are approximate with a mid-tier model.)*

---

## 🔧 API Endpoints

| Endpoint   | Method | Description                    |
|------------|--------|--------------------------------|
| `/`        | GET    | Environment info               |
| `/health`  | GET    | Health check (returns 200)     |
| `/reset`   | POST   | Reset environment for new task |
| `/step`    | POST   | Execute an agent action        |
| `/state`   | GET    | Current environment state      |
| `/schema`  | GET    | Action/Observation JSON schemas|
| `/tasks`   | GET    | List available tasks           |

---

## 🏗️ Reward Design

The reward function provides **meaningful signal at every step**:

| Action                           | Reward     |
|----------------------------------|------------|
| Correct classification           | +0.20      |
| Close classification (related)   | +0.08      |
| Wrong classification             | −0.05      |
| Correct priority                 | +0.15      |
| Priority off by 1                | +0.07      |
| Correct department routing       | +0.15      |
| Good response draft (≥60% keys)  | +0.30      |
| Correct escalation               | +0.05      |
| Unnecessary escalation           | −0.10      |
| Re-processing same email         | −0.15      |
| Invalid email_id                 | −0.10      |

---

## 📜 License

MIT

## 👥 Team

**Ctrl-Alt-Defeat**
- Haraprasad Hota (Team Lead)
- Subhendu Samal
- Ashutosh Panigrahi
