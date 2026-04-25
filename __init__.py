# Email Triage Environment — Root Package
"""
OpenEnv-compliant Email Triage RL Environment.
Powered by openenv-core (https://github.com/meta-pytorch/OpenEnv).

Public API:
    # Models
    from email_triage_env import EmailAction, EmailObservation, State, EmailData

    # Environment (server-side)
    from email_triage_env import EmailTriageEnvironment

    # Client (connect to running server over WebSocket)
    from email_triage_env import EmailTriageClient

Quick start:
    # Standalone (no server needed)
    from email_triage_env import EmailTriageEnvironment
    env = EmailTriageEnvironment()
    obs = env.reset(task_id='easy')

    # Remote client (openenv-core WebSocket protocol)
    from email_triage_env import EmailTriageClient
    async with EmailTriageClient.hf_space() as client:
        result = await client.reset(task_id='easy')
"""

from models import EmailAction, EmailObservation, State, EmailData
from server.environment import EmailTriageEnvironment
from client import EmailTriageClient

__all__ = [
    # Models
    "EmailAction",
    "EmailObservation",
    "State",
    "EmailData",
    # Environment
    "EmailTriageEnvironment",
    # Client
    "EmailTriageClient",
]

__version__ = "2.0.0"
