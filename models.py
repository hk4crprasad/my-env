"""
Pydantic models for the Email Triage Environment.

Defines typed Action, Observation, and State models following the OpenEnv spec.
Falls back to standalone Pydantic models if openenv-core is not installed.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# OpenEnv base classes – import from openenv-core if available, else define
# minimal compatible stubs so the environment works stand-alone too.
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_server.types import Action as _BaseAction
    from openenv.core.env_server.types import Observation as _BaseObservation
    from openenv.core.env_server.types import State as _BaseState

    HAS_OPENENV = True
except ImportError:
    HAS_OPENENV = False

    class _BaseAction(BaseModel):
        """Minimal Action stub matching openenv-core interface."""

        model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class _BaseObservation(BaseModel):
        """Minimal Observation stub matching openenv-core interface."""

        model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )
        done: bool = Field(default=False)
        reward: Optional[float] = Field(default=None)
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class _BaseState(BaseModel):
        """Minimal State stub matching openenv-core interface."""

        model_config = ConfigDict(
            extra="allow",
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )
        episode_id: Optional[str] = Field(default=None)
        step_count: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# Domain-specific helper models
# ---------------------------------------------------------------------------

class EmailData(BaseModel):
    """Represents a single email in the inbox."""

    model_config = ConfigDict(extra="forbid")

    email_id: str = Field(description="Unique identifier for this email")
    sender: str = Field(description="Sender email address")
    sender_name: str = Field(default="", description="Sender display name")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body text")
    timestamp: str = Field(description="ISO-format timestamp")
    has_attachment: bool = Field(default=False)
    is_reply: bool = Field(default=False)
    thread_id: Optional[str] = Field(default=None, description="Thread ID for reply chains")
    labels: List[str] = Field(default_factory=list, description="Existing labels if any")


class EmailGroundTruth(BaseModel):
    """Ground truth labels for a single email (used by graders)."""

    model_config = ConfigDict(extra="forbid")

    email_id: str
    category: str = Field(description="spam | billing | technical | general | urgent")
    priority: int = Field(ge=1, le=5, description="1 = critical … 5 = low")
    department: str = Field(description="engineering | billing | support | management")
    requires_response: bool = Field(default=False)
    expected_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords expected in a good response draft",
    )


class InboxStats(BaseModel):
    """Summary statistics of the current inbox."""

    total: int = 0
    unprocessed: int = 0
    processed: int = 0
    by_category: Dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# OpenEnv Action model — what the agent sends
# ---------------------------------------------------------------------------

class EmailAction(_BaseAction):
    """Action taken by the agent on a single email.

    The agent must specify which email to act on and provide at least
    a classification. Priority, routing, and response drafts are optional
    but improve the final score.
    """

    email_id: str = Field(description="ID of the email to act on")
    category: Optional[str] = Field(
        default=None,
        description="Classification: spam | billing | technical | general | urgent",
    )
    priority: Optional[int] = Field(
        default=None,
        ge=1,
        le=5,
        description="Priority 1 (critical) to 5 (low)",
    )
    department: Optional[str] = Field(
        default=None,
        description="Route to: engineering | billing | support | management",
    )
    response_draft: Optional[str] = Field(
        default=None,
        description="Draft reply text (for emails requiring a response)",
    )
    escalate: bool = Field(
        default=False,
        description="Flag for escalation to management",
    )


# ---------------------------------------------------------------------------
# OpenEnv Observation model — what the agent receives
# ---------------------------------------------------------------------------

class EmailObservation(_BaseObservation):
    """Observation returned to the agent after each step.

    Contains the current inbox state, task info, and feedback on the last
    action taken.
    """

    emails: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of emails currently in the inbox (unprocessed)",
    )
    inbox_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Inbox summary statistics",
    )
    task_id: str = Field(default="", description="Current task identifier")
    task_description: str = Field(default="", description="Human-readable task instructions")
    action_feedback: Optional[str] = Field(
        default=None,
        description="Feedback on the last action (e.g. 'correct', 'wrong category')",
    )
    step_reward: float = Field(
        default=0.0,
        description="Reward earned from the last action",
    )
    cumulative_reward: float = Field(
        default=0.0,
        description="Total reward accumulated so far",
    )
    steps_remaining: Optional[int] = Field(
        default=None,
        description="Steps remaining before episode truncation",
    )


# ---------------------------------------------------------------------------
# Re-export base State as-is (uses extra='allow' so can hold custom fields)
# ---------------------------------------------------------------------------

State = _BaseState


# ---------------------------------------------------------------------------
# Valid value sets (for validation & prompting)
# ---------------------------------------------------------------------------

VALID_CATEGORIES = {"spam", "billing", "technical", "general", "urgent"}
VALID_DEPARTMENTS = {"engineering", "billing", "support", "management"}
VALID_PRIORITIES = {1, 2, 3, 4, 5}
