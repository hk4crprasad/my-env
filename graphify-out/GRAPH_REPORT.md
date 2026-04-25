# Graph Report - .  (2026-04-25)

## Corpus Check
- Corpus is ~24,316 words - fits in a single context window. You may not need a graph.

## Summary
- 295 nodes · 597 edges · 13 communities detected
- Extraction: 58% EXTRACTED · 42% INFERRED · 0% AMBIGUOUS · INFERRED: 249 edges (avg confidence: 0.58)
- Token cost: 5,000 input · 2,400 output

## Community Hubs (Navigation)
- [[_COMMUNITY_API Server & Session Management|API Server & Session Management]]
- [[_COMMUNITY_Email Generation & Inbox Simulation|Email Generation & Inbox Simulation]]
- [[_COMMUNITY_Reward Rubric & Demo Interface|Reward Rubric & Demo Interface]]
- [[_COMMUNITY_Analytics, Database & Persistence|Analytics, Database & Persistence]]
- [[_COMMUNITY_Environment Lifecycle & Plots|Environment Lifecycle & Plots]]
- [[_COMMUNITY_GRPO Training Pipeline|GRPO Training Pipeline]]
- [[_COMMUNITY_Documentation & Design Rationale|Documentation & Design Rationale]]
- [[_COMMUNITY_Curriculum & Task Definitions|Curriculum & Task Definitions]]
- [[_COMMUNITY_Reward Separation Evidence|Reward Separation Evidence]]
- [[_COMMUNITY_Training Performance Results|Training Performance Results]]
- [[_COMMUNITY_Dependency Management|Dependency Management]]
- [[_COMMUNITY_Package Init|Package Init]]
- [[_COMMUNITY_Server Package Init|Server Package Init]]

## God Nodes (most connected - your core abstractions)
1. `EmailTriageEnvironment` - 63 edges
2. `EmailAction` - 40 edges
3. `EmailObservation` - 40 edges
4. `EmailGroundTruth` - 36 edges
5. `TaskDefinition` - 19 edges
6. `EmailData` - 17 edges
7. `InboxStats` - 12 edges
8. `EmailTriageClient` - 12 edges
9. `DatabaseManager` - 12 edges
10. `_InMemoryStore` - 11 edges

## Surprising Connections (you probably didn't know these)
- `Reward shaping for the Email Triage Environment.  Each dimension is its own inde` --uses--> `EmailGroundTruth`  [INFERRED]
  server/reward.py → models.py
- `Independent reward for email category classification.` --uses--> `EmailGroundTruth`  [INFERRED]
  server/reward.py → models.py
- `Independent reward for priority assignment.` --uses--> `EmailGroundTruth`  [INFERRED]
  server/reward.py → models.py
- `Independent reward for department routing.` --uses--> `EmailGroundTruth`  [INFERRED]
  server/reward.py → models.py
- `Independent reward for response draft quality (keyword-based, no LLM).` --uses--> `EmailGroundTruth`  [INFERRED]
  server/reward.py → models.py

## Hyperedges (group relationships)
- **Anti-Reward-Hacking Mechanisms** — readme_format_gate, readme_deduplication_penalty, readme_hidden_keywords, readme_seven_reward_components [EXTRACTED 0.95]
- **Five Triage Decision Dimensions** — readme_email_classification, readme_email_priority, readme_email_routing, readme_phishing_detection, readme_thread_context [EXTRACTED 0.90]
- **Full Training Pipeline** — readme_grpo_training, readme_curriculum_learning, readme_trained_adapter, readme_qwen_base_model [EXTRACTED 0.90]
- **GRPO Training Improvement Evidence** — score_comparison_chart, training_curve_chart, dimension_breakdown_chart, reward_spread_chart [INFERRED 0.90]

## Communities

### Community 0 - "API Server & Session Management"
Cohesion: 0.06
Nodes (48): _create_session(), _get_session(), get_state(), health(), HealthResponse, inference_runs(), leaderboard(), FastAPI application for the Email Triage Environment.  OpenEnv-core endpoints (W (+40 more)

### Community 1 - "Email Generation & Inbox Simulation"
Cohesion: 0.1
Nodes (40): _easy_emails(), generate_emails(), _hard_emails(), _make_id(), _medium_emails(), Deterministic email generator for the Email Triage Environment.  Provides realis, Generate a deterministic email ID from seed and index., Generate emails and ground truth for the specified task.      Args:         task (+32 more)

### Community 2 - "Reward Rubric & Demo Interface"
Cohesion: 0.09
Nodes (35): Return all reward rubric definitions.      Each reward component is independent, rubric(), build_ui(), format_email(), main(), new_env(), Gradio demo for the Email Triage RL Environment.  Lets a human (or LLM) play thr, Start a new episode and return the first email + status. (+27 more)

### Community 3 - "Analytics, Database & Persistence"
Cohesion: 0.07
Nodes (16): analytics(), Per-task aggregated statistics (avg/best/worst score, run count)., startup(), DatabaseManager, _InMemoryStore, MongoDB integration for the Email Triage Environment.  Provides persistent stora, Connect to MongoDB. Falls back to in-memory on failure., Create required indexes (idempotent). (+8 more)

### Community 4 - "Environment Lifecycle & Plots"
Cohesion: 0.09
Nodes (29): shutdown(), plot_dimension_breakdown(), plot_reward_spread(), plot_score_comparison(), plot_training_curve(), Generate baseline & training-evidence plots for the README.  These plots are com, Bar chart: baseline vs. trained final scores by task., Show which dimensions improved most after training. (+21 more)

### Community 5 - "GRPO Training Pipeline"
Cohesion: 0.13
Nodes (23): build_dataset(), evaluate_model(), format_email_prompt(), load_model(), main(), _parse_action(), parse_args(), GRPO Training Script — Email Triage RL Environment ============================= (+15 more)

### Community 6 - "Documentation & Design Rationale"
Cohesion: 0.09
Nodes (23): CLAUDE.md Project Guide, Server Startup Command, Validation Script Command, Curriculum Learning (Easy→Medium→Hard), Deduplication Penalty (-0.15), Email Classification (spam/billing/technical/general/urgent), Email Priority Assignment (1-5), Email Department Routing (+15 more)

### Community 7 - "Curriculum & Task Definitions"
Cohesion: 0.2
Nodes (10): curriculum(), Return the curriculum progression (easy → medium → hard)., root(), get_curriculum_order(), get_task(), list_task_ids(), Task definitions for the Email Triage Environment.  Three difficulty levels form, Retrieve a task definition by ID. (+2 more)

### Community 8 - "Reward Separation Evidence"
Cohesion: 0.29
Nodes (7): Per-Dimension Improvement Chart (Medium Task), Classification Accuracy: 0.48→0.72 (+0.24), Priority Accuracy: 0.31→0.55 (+0.24), Routing Accuracy: 0.29→0.50 (+0.21), Reward Component Separation Chart, HARD Task Reward Separation, Perfect vs Random Action Comparison

### Community 9 - "Training Performance Results"
Cohesion: 0.47
Nodes (6): Baseline vs GRPO Score Comparison, Easy Task: Baseline=0.60 → GRPO=0.92 (+0.32), Hard Task: Baseline=0.29 → GRPO=0.51 (+0.22), Medium Task: Baseline=0.38 → GRPO=0.64 (+0.26), GRPO Training Reward Curve, Reward Improvement Over Training Steps

### Community 10 - "Dependency Management"
Cohesion: 1.0
Nodes (2): Runtime Dependencies, Training Dependencies (TRL, Unsloth)

### Community 11 - "Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 12 - "Server Package Init"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **50 isolated node(s):** `Pydantic models for the Email Triage Environment.  Defines typed Action, Observa`, `Minimal Action stub matching openenv-core interface.`, `Minimal Observation stub matching openenv-core interface.`, `Minimal State stub matching openenv-core interface.`, `Represents a single email in the inbox.` (+45 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Dependency Management`** (2 nodes): `Runtime Dependencies`, `Training Dependencies (TRL, Unsloth)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Server Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `EmailTriageEnvironment` connect `Reward Rubric & Demo Interface` to `API Server & Session Management`, `Email Generation & Inbox Simulation`, `Analytics, Database & Persistence`, `Environment Lifecycle & Plots`, `GRPO Training Pipeline`, `Curriculum & Task Definitions`?**
  _High betweenness centrality (0.388) - this node is a cross-community bridge._
- **Why does `EmailGroundTruth` connect `Email Generation & Inbox Simulation` to `API Server & Session Management`, `Reward Rubric & Demo Interface`, `Environment Lifecycle & Plots`?**
  _High betweenness centrality (0.141) - this node is a cross-community bridge._
- **Why does `EmailAction` connect `API Server & Session Management` to `Email Generation & Inbox Simulation`, `Reward Rubric & Demo Interface`, `Analytics, Database & Persistence`, `Curriculum & Task Definitions`?**
  _High betweenness centrality (0.117) - this node is a cross-community bridge._
- **Are the 54 inferred relationships involving `EmailTriageEnvironment` (e.g. with `Gradio demo for the Email Triage RL Environment.  Lets a human (or LLM) play thr` and `Start a new episode and return the first email + status.`) actually correct?**
  _`EmailTriageEnvironment` has 54 INFERRED edges - model-reasoned connections that need verification._
- **Are the 37 inferred relationships involving `EmailAction` (e.g. with `EmailTriageClient` and `_HTTPStepResult`) actually correct?**
  _`EmailAction` has 37 INFERRED edges - model-reasoned connections that need verification._
- **Are the 37 inferred relationships involving `EmailObservation` (e.g. with `EmailTriageClient` and `_HTTPStepResult`) actually correct?**
  _`EmailObservation` has 37 INFERRED edges - model-reasoned connections that need verification._
- **Are the 33 inferred relationships involving `EmailGroundTruth` (e.g. with `Reward shaping for the Email Triage Environment.  Each dimension is its own inde` and `Independent reward for email category classification.`) actually correct?**
  _`EmailGroundTruth` has 33 INFERRED edges - model-reasoned connections that need verification._