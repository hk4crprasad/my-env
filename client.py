"""
client.py — Email Triage Environment Client
=============================================
OpenEnv Hackathon 2026 · Team Ctrl-Alt-Defeat

Official EnvClient implementation using openenv-core.
Connects to the Email Triage server over WebSocket/HTTP using the
Meta OpenEnv protocol — compatible with TRL, Unsloth, ART, and Oumi.

Usage (async):
    from client import EmailTriageClient
    from models import EmailAction

    async with EmailTriageClient(base_url="http://localhost:7860") as client:
        result = await client.reset(task_id="easy", seed=42)
        while not result.observation.done:
            action = EmailAction(
                email_id=result.observation.emails[0]["email_id"],
                category="billing",
                priority=3,
                department="billing",
            )
            result = await client.step(action)
        print(result.observation.metadata["grading"])

Usage (sync):
    with EmailTriageClient(base_url="http://localhost:7860").sync() as client:
        result = client.reset(task_id="easy")
        result = client.step(action)

Usage with HF Space:
    client = EmailTriageClient(
        base_url="https://hk4crprasad-email-triage-env.hf.space"
    )

Links:
    HF Space  : https://huggingface.co/spaces/Hk4crprasad/email-triage-env
    GitHub    : https://github.com/hk4crprasad/my-env
"""

from __future__ import annotations

from typing import Optional, Type

try:
    from openenv.core import EnvClient, SyncEnvClient
    from openenv.core.env_server.types import State
    _HAS_OPENENV_CORE = True
except ImportError:
    _HAS_OPENENV_CORE = False

from models import EmailAction, EmailObservation

# Default HF Space URL
HF_SPACE_URL = "https://hk4crprasad-email-triage-env.hf.space"
LOCAL_URL     = "http://localhost:7860"


if _HAS_OPENENV_CORE:
    class EmailTriageClient(EnvClient[EmailAction, EmailObservation, State]):
        """
        OpenEnv-core EnvClient for the Email Triage Environment.

        Connects over WebSocket to the server (local or HF Space) and
        communicates using the standard OpenEnv protocol.

        Inherits from openenv-core EnvClient:
          - reset(**kwargs)  → StepResult[EmailObservation]
          - step(action)     → StepResult[EmailObservation]
          - state()          → State
          - .sync()          → SyncEnvClient wrapper

        Example:
            async with EmailTriageClient() as client:
                result = await client.reset(task_id="easy", seed=42)
                obs = result.observation
                action = EmailAction(
                    email_id=obs.emails[0]["email_id"],
                    category="spam",
                    priority=5,
                    department="support",
                )
                result = await client.step(action)
                print(f"Reward: {result.reward}, Done: {result.done}")
        """

        _action_cls = EmailAction
        _observation_cls = EmailObservation
        _state_cls = State

        def __init__(
            self,
            base_url: str = LOCAL_URL,
            connect_timeout_s: float = 30.0,
            message_timeout_s: float = 120.0,
        ):
            super().__init__(
                base_url=base_url,
                connect_timeout_s=connect_timeout_s,
                message_timeout_s=message_timeout_s,
            )

        # ── Convenience class methods ──────────────────────────────────────

        @classmethod
        def local(cls) -> "EmailTriageClient":
            """Connect to a locally running server."""
            return cls(base_url=LOCAL_URL)

        @classmethod
        def hf_space(cls) -> "EmailTriageClient":
            """Connect to the deployed HF Space."""
            return cls(base_url=HF_SPACE_URL)

else:
    # Fallback: minimal HTTP-based client (no openenv-core required)
    import requests

    class EmailTriageClient:  # type: ignore[no-redef]
        """
        Fallback HTTP client for environments without openenv-core.
        Uses the REST API (POST /reset, POST /step, GET /state).

        Install openenv-core for the full WebSocket-based client:
            pip install openenv-core>=0.2.0
        """

        def __init__(self, base_url: str = LOCAL_URL, **_):
            self.base_url = base_url.rstrip("/")
            self._session_id: Optional[str] = None

        def reset(self, task_id: str = "easy", seed: int = 42, **kwargs):
            resp = requests.post(
                f"{self.base_url}/reset",
                json={"task_id": task_id, "seed": seed},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            self._session_id = data.get("session_id")
            return _HTTPStepResult(data)

        def step(self, action: dict, **kwargs):
            payload = action if isinstance(action, dict) else action.model_dump()
            resp = requests.post(
                f"{self.base_url}/step",
                json={"action": payload, "session_id": self._session_id},
                timeout=30,
            )
            resp.raise_for_status()
            return _HTTPStepResult(resp.json())

        def state(self):
            resp = requests.get(
                f"{self.base_url}/state",
                params={"session_id": self._session_id},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def sync(self):
            return self


    class _HTTPStepResult:
        """Minimal StepResult wrapper matching openenv-core interface."""
        def __init__(self, data: dict):
            self._data = data
            obs_dict = data.get("observation", data)
            self.observation = _DictObservation(obs_dict)
            self.reward = data.get("reward", 0.0)
            self.done = data.get("done", False)

        def __repr__(self):
            return (
                f"StepResult(reward={self.reward}, done={self.done}, "
                f"emails={len(self.observation.emails)})"
            )


    class _DictObservation:
        """Dict-backed observation with attribute access."""
        def __init__(self, d: dict):
            for k, v in d.items():
                setattr(self, k, v)
            self.emails = d.get("emails", [])
            self.done = d.get("done", False)
            self.reward = d.get("reward", 0.0)
            self.metadata = d.get("metadata", {})


# ── Quick demo ─────────────────────────────────────────────────────────────

def _demo():
    """Quick test: connect to local server and run one easy episode."""
    import asyncio

    async def _run():
        print("Connecting to local Email Triage server...")
        async with EmailTriageClient.local() as client:
            result = await client.reset(task_id="easy", seed=42)
            print(f"Reset: {len(result.observation.emails)} emails in inbox")
            step = 0
            while not result.observation.done and step < 10:
                email = result.observation.emails[0] if result.observation.emails else None
                if not email:
                    break
                action = EmailAction(
                    email_id=email["email_id"],
                    category="general",
                    priority=3,
                    department="support",
                )
                result = await client.step(action)
                print(f"  Step {step+1}: reward={result.reward:+.3f}, done={result.done}")
                step += 1
            grading = result.observation.metadata.get("grading", {})
            print(f"\nFinal score: {grading.get('final_score', 'N/A')}")

    if _HAS_OPENENV_CORE:
        asyncio.run(_run())
    else:
        print("Demo requires openenv-core: pip install openenv-core>=0.2.0")
        print("Or run the fallback HTTP demo against a local server.")


if __name__ == "__main__":
    _demo()
