# Email Triage Environment — Root Package
"""
OpenEnv-compliant Email Triage Environment.

Public API:
    from email_triage_env import EmailAction, EmailObservation, EmailTriageEnvironment
"""

from models import EmailAction, EmailObservation, State, EmailData

__all__ = ["EmailAction", "EmailObservation", "State", "EmailData"]
