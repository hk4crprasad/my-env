"""
Multi-Agent Email Triage Environment — Theme #1: Multi-Agent Interactions

Architecture: 3 sequential specialist agents collaborate on each email:

  Analyst  →  Router  →  Responder
     ↓            ↓           ↓
  category     department   draft
  priority     escalation   tone
  signals      reasoning    coalition

Each agent sees the outputs of all upstream agents (theory-of-mind context).
The environment rewards both individual accuracy AND inter-agent coordination:
  • Coordination reward  — analyst/router decisions are semantically aligned
  • Theory-of-mind score — router demonstrably uses analyst signals
  • Coalition score      — responder draft is coherent with both upstream agents

This creates genuine multi-agent RL: individually-correct but inconsistent
decisions are penalised, pushing the policy towards coordinated team behaviour.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional
from uuid import uuid4

from models import (
    MultiAgentAction,
    MultiAgentObservation,
    AgentFeedback,
    State,
    VALID_CATEGORIES,
    VALID_DEPARTMENTS,
)
from server.email_generator import generate_emails
from server.tasks import get_task, TASKS
from server.reward import (
    compute_multi_agent_reward,
    reward_reprocess_penalty,
)


# ── System prompts for each specialist agent ─────────────────────────────────
#
# Each agent is the SAME Llama-3.2-1B-Instruct model but with a different
# system prompt. This teaches role-conditioned behaviour and theory-of-mind.

ANALYST_SYSTEM_PROMPT = """You are the Analyst agent in a 3-agent email triage team.

Your role: read the email carefully and output a JSON analysis.
You are the FIRST agent — other agents will use your output, so be accurate.

Respond with ONLY this JSON:
{
  "category": "<spam|billing|technical|general|urgent>",
  "priority": <1-5>,
  "signals": ["<key signal 1>", "<key signal 2>"],
  "confidence": <0.0-1.0>
}

Categories: spam=phishing/unsolicited, billing=payments/invoices,
technical=bugs/API/code, general=inquiries, urgent=outages/security/legal

Signals: list the 2-3 most important text signals you used to decide
(e.g. ["suspicious domain", "bit.ly redirect", "fake urgency"])

Priority: 1=critical(system down/breach), 2=high, 3=medium, 4=low, 5=spam/auto"""


ROUTER_SYSTEM_PROMPT = """You are the Router agent in a 3-agent email triage team.

Your role: given the email AND the Analyst's findings, decide routing and escalation.
You MUST use the Analyst's category and signals to inform your routing decision
(this is scored as theory-of-mind — ignoring the Analyst will cost you points).

Respond with ONLY this JSON:
{
  "department": "<engineering|billing|support|management>",
  "escalate": <true|false>,
  "analyst_agreement": <true|false>,
  "routing_reason": "<one-line reason referencing the analyst's signals>"
}

Department guide:
- engineering: technical bugs, API issues, security vulnerabilities
- billing: payment disputes, invoices, refunds
- support: general questions, account issues, spam handling
- management: legal threats, GDPR, compliance, critical escalations

Escalate only for: priority=1 (critical outage/breach) OR management-bound issues."""


RESPONDER_SYSTEM_PROMPT = """You are the Responder agent in a 3-agent email triage team.

Your role: given the email AND both the Analyst's classification AND the Router's
decision, draft a response. Your draft MUST be consistent with both upstream agents:
reference the category domain AND the department context (coalition scoring).

Respond with ONLY this JSON:
{
  "response_draft": "<your draft reply, or null if not applicable>",
  "tone": "<formal|empathetic|technical|urgent>"
}

Write a response ONLY for:
- billing complaints, account issues, technical incidents, legal/compliance matters
- anything the Analyst classified as billing, technical, or urgent

For spam: set response_draft to null.
For general inquiries: brief, polite acknowledgment.
Keep drafts concise (2-4 sentences). Reference the specific issue."""


def build_analyst_prompt(email: Dict[str, Any], task_description: str) -> str:
    """Build the prompt shown to the Analyst agent."""
    return (
        f"TASK: {task_description[:150]}\n\n"
        f"Email ID: {email['email_id']}\n"
        f"From: {email.get('sender_name', '')} <{email.get('sender', '')}>\n"
        f"Subject: {email.get('subject', '')}\n"
        f"Is Reply: {email.get('is_reply', False)}\n"
        f"Has Attachment: {email.get('has_attachment', False)}\n\n"
        f"Body:\n{email.get('body', '(empty)')}"
    )


def build_router_prompt(
    email: Dict[str, Any],
    task_description: str,
    analyst_output: Dict[str, Any],
) -> str:
    """Build the prompt shown to the Router agent (includes Analyst's output)."""
    return (
        f"TASK: {task_description[:150]}\n\n"
        f"Email ID: {email['email_id']}\n"
        f"From: {email.get('sender_name', '')} <{email.get('sender', '')}>\n"
        f"Subject: {email.get('subject', '')}\n\n"
        f"Body:\n{email.get('body', '(empty)')}\n\n"
        f"--- Analyst's findings ---\n"
        f"category   : {analyst_output.get('category', '?')}\n"
        f"priority   : {analyst_output.get('priority', '?')}\n"
        f"signals    : {analyst_output.get('signals', [])}\n"
        f"confidence : {analyst_output.get('confidence', 1.0):.2f}\n"
        f"Use these findings to make your routing decision."
    )


def build_responder_prompt(
    email: Dict[str, Any],
    task_description: str,
    analyst_output: Dict[str, Any],
    router_output: Dict[str, Any],
) -> str:
    """Build the prompt shown to the Responder agent (full context)."""
    return (
        f"TASK: {task_description[:150]}\n\n"
        f"Email ID: {email['email_id']}\n"
        f"From: {email.get('sender_name', '')} <{email.get('sender', '')}>\n"
        f"Subject: {email.get('subject', '')}\n\n"
        f"Body:\n{email.get('body', '(empty)')}\n\n"
        f"--- Analyst's findings ---\n"
        f"category: {analyst_output.get('category', '?')}  "
        f"priority: {analyst_output.get('priority', '?')}\n"
        f"signals: {analyst_output.get('signals', [])}\n\n"
        f"--- Router's decision ---\n"
        f"department : {router_output.get('department', '?')}\n"
        f"escalate   : {router_output.get('escalate', False)}\n"
        f"reason     : {router_output.get('routing_reason', 'no reason given')}\n\n"
        f"Draft a response consistent with both agents' decisions above."
    )


# ─────────────────────────────────────────────────────────────────────────────

class MultiAgentTriageEnvironment:
    """OpenEnv-compatible multi-agent environment.

    Three specialist agents (Analyst, Router, Responder) collaborate on each
    email step.  The environment returns per-agent feedback AND coordination
    scores that reward inter-agent consistency.

    Compatible with the same OpenEnv reset/step/state interface.
    """

    def __init__(self) -> None:
        self._emails:        List[Any] = []
        self._ground_truths: Dict[str, Any] = {}
        self._processed:     set = set()
        self._state:         State = State()
        self._task_id:       str = "easy"
        self._task_description: str = ""
        self._cumulative:    float = 0.0

    # ── OpenEnv interface ─────────────────────────────────────────────────

    def reset(
        self,
        task_id: str = "easy",
        seed: int = 42,
        episode_id: Optional[str] = None,
    ) -> MultiAgentObservation:
        task = get_task(task_id)
        emails, gts = generate_emails(task_id, seed)

        self._emails        = list(emails)
        self._ground_truths = {gt.email_id: gt for gt in gts}
        self._processed     = set()
        self._task_id       = task_id
        self._task_description = task.description
        self._cumulative    = 0.0

        eid = episode_id or str(uuid4())
        self._state = State(
            episode_id=eid,
            step_count=0,
        )

        return self._make_obs(
            step_reward=0.0,
            coord_reward=0.0,
            agent_feedback=[],
            action_feedback="Multi-agent episode started. Three agents ready.",
        )

    def step(self, action: MultiAgentAction) -> MultiAgentObservation:
        """Process one email with the 3-agent team."""
        email_id = action.email_id
        gt       = self._ground_truths.get(email_id)

        # ── Re-processing guard ───────────────────────────────────────────
        if email_id in self._processed:
            pen = reward_reprocess_penalty(already_processed=True)
            self._cumulative += pen
            self._state.step_count += 1
            return self._make_obs(
                step_reward=pen,
                coord_reward=0.0,
                agent_feedback=[AgentFeedback(
                    agent="all",
                    reward=pen,
                    feedback="⚠ This email was already processed. −0.15 penalty.",
                )],
                action_feedback=f"Penalty: email {email_id} already processed.",
            )

        if gt is None:
            self._state.step_count += 1
            return self._make_obs(
                step_reward=-0.15,
                coord_reward=0.0,
                agent_feedback=[AgentFeedback(
                    agent="all", reward=-0.15,
                    feedback=f"✗ Unknown email_id '{email_id}'",
                )],
                action_feedback=f"✗ Unknown email_id '{email_id}'. Check the inbox.",
            )

        task    = TASKS[self._task_id]
        valid   = {e.email_id for e in self._emails}

        # ── Compute full multi-agent reward ───────────────────────────────
        total, coord, per_agent, feedback = compute_multi_agent_reward(
            email_id          = email_id,
            analyst_category  = action.analyst.category,
            analyst_priority  = action.analyst.priority,
            analyst_signals   = action.analyst.signals,
            router_department = action.router.department,
            router_escalate   = action.router.escalate,
            router_reason     = action.router.routing_reason,
            response_draft    = action.responder.response_draft,
            ground_truth      = gt,
            requires_response = task.requires_response,
            valid_email_ids   = valid,
        )

        self._processed.add(email_id)
        self._cumulative += total
        self._state.step_count += 1

        # Remove processed email from inbox
        self._emails = [e for e in self._emails if e.email_id != email_id]

        agent_fb = [
            AgentFeedback(agent="analyst",   reward=per_agent["analyst"],
                          feedback=f"cat={action.analyst.category} pri={action.analyst.priority}"),
            AgentFeedback(agent="router",    reward=per_agent["router"],
                          feedback=f"dept={action.router.department} esc={action.router.escalate}"),
            AgentFeedback(agent="responder", reward=per_agent["responder"],
                          feedback=f"draft={'yes' if action.responder.response_draft else 'none'}"),
            AgentFeedback(agent="coordination", reward=per_agent["coordination"],
                          feedback=feedback),
        ]

        done = (not self._emails) or (
            self._state.step_count >= TASKS[self._task_id].max_steps
        )

        return self._make_obs(
            step_reward=total,
            coord_reward=coord,
            agent_feedback=agent_fb,
            action_feedback=feedback,
            done=done,
        )

    @property
    def state(self) -> State:
        return self._state

    # ── Internal helpers ──────────────────────────────────────────────────

    def _make_obs(
        self,
        step_reward: float,
        coord_reward: float,
        agent_feedback: List[AgentFeedback],
        action_feedback: str,
        done: bool = False,
    ) -> MultiAgentObservation:
        remaining = max(0, TASKS[self._task_id].max_steps - self._state.step_count)
        return MultiAgentObservation(
            emails=[e.model_dump() for e in self._emails],
            inbox_stats={
                "total":       len(self._ground_truths),
                "processed":   len(self._processed),
                "unprocessed": len(self._emails),
            },
            task_id          = self._task_id,
            task_description = self._task_description,
            agent_feedback   = agent_feedback,
            coordination_reward  = coord_reward,
            theory_of_mind_score = 0.0,
            step_reward      = step_reward,
            cumulative_reward= self._cumulative,
            steps_remaining  = remaining,
            action_feedback  = action_feedback,
            done             = done,
            reward           = step_reward,
            metadata         = {
                "step_count":  self._state.step_count,
                "processed":   list(self._processed),
                "per_agent":   {fb.agent: fb.reward for fb in agent_feedback},
            },
        )


# ── Prompt builders for training / inference ─────────────────────────────────
# These are exported so train.py and demo.py can use them directly.

def get_agent_prompts(
    email: Dict[str, Any],
    task_description: str,
    analyst_out: Optional[Dict[str, Any]] = None,
    router_out:  Optional[Dict[str, Any]] = None,
) -> Dict[str, tuple]:
    """Return (system_prompt, user_prompt) for each agent.

    Call with analyst_out=None, router_out=None to get the Analyst prompt.
    Call with analyst_out filled to get the Router prompt.
    Call with both filled to get the Responder prompt.
    """
    prompts: Dict[str, tuple] = {}

    prompts["analyst"] = (
        ANALYST_SYSTEM_PROMPT,
        build_analyst_prompt(email, task_description),
    )

    if analyst_out is not None:
        prompts["router"] = (
            ROUTER_SYSTEM_PROMPT,
            build_router_prompt(email, task_description, analyst_out),
        )

    if analyst_out is not None and router_out is not None:
        prompts["responder"] = (
            RESPONDER_SYSTEM_PROMPT,
            build_responder_prompt(email, task_description, analyst_out, router_out),
        )

    return prompts
