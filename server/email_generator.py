"""
Deterministic email generator for the Email Triage Environment.

Provides realistic synthetic emails with ground-truth labels for each
difficulty level. Uses seed-based selection for reproducibility.
"""

from __future__ import annotations

import hashlib
from typing import List, Tuple

from models import EmailData, EmailGroundTruth


def _make_id(seed: int, index: int) -> str:
    """Generate a deterministic email ID from seed and index."""
    raw = f"{seed}-{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ═══════════════════════════════════════════════════════════════════════════
#  EASY TASK EMAILS  —  5 clear-cut emails
# ═══════════════════════════════════════════════════════════════════════════

def _easy_emails(seed: int = 42) -> List[Tuple[EmailData, EmailGroundTruth]]:
    emails = [
        (
            EmailData(
                email_id=_make_id(seed, 0),
                sender="winner@prize-giveaway-2024.com",
                sender_name="Lucky Draw Committee",
                subject="🎉 CONGRATULATIONS! You've been selected for $1,000,000!",
                body=(
                    "Dear Lucky Winner,\n\n"
                    "We are pleased to inform you that your email address has been "
                    "randomly selected in our annual global prize draw! You have won "
                    "ONE MILLION DOLLARS ($1,000,000.00 USD).\n\n"
                    "To claim your prize, simply click the link below and provide "
                    "your bank details for immediate transfer:\n"
                    "http://totally-legit-prizes.com/claim?id=XK129\n\n"
                    "Act now! This offer expires in 24 hours!\n\n"
                    "Congratulations once again!\n"
                    "International Prize Commission"
                ),
                timestamp="2024-11-15T08:23:00Z",
                has_attachment=False,
                is_reply=False,
            ),
            EmailGroundTruth(
                email_id=_make_id(seed, 0),
                category="spam",
                priority=5,
                department="support",
                requires_response=False,
                expected_keywords=[],
            ),
        ),
        (
            EmailData(
                email_id=_make_id(seed, 1),
                sender="sarah.chen@techstartup.io",
                sender_name="Sarah Chen",
                subject="Invoice #INV-2024-0892 — Charge Discrepancy",
                body=(
                    "Hi Billing Team,\n\n"
                    "I've noticed a discrepancy on my latest invoice (INV-2024-0892). "
                    "I was quoted $199/month for the Pro plan, but the invoice shows "
                    "$299. Could you please look into this and issue a corrected invoice?\n\n"
                    "I have attached a screenshot of the original quote for reference.\n\n"
                    "Thanks,\nSarah Chen\nCTO, TechStartup.io"
                ),
                timestamp="2024-11-15T09:45:00Z",
                has_attachment=True,
                is_reply=False,
            ),
            EmailGroundTruth(
                email_id=_make_id(seed, 1),
                category="billing",
                priority=3,
                department="billing",
                requires_response=False,
                expected_keywords=[],
            ),
        ),
        (
            EmailData(
                email_id=_make_id(seed, 2),
                sender="devops@clientcorp.com",
                sender_name="ClientCorp DevOps",
                subject="Bug Report: Dashboard crashes on Firefox 115",
                body=(
                    "Environment: Firefox 115.0 on Ubuntu 22.04\n\n"
                    "Steps to reproduce:\n"
                    "1. Log in to the analytics dashboard\n"
                    "2. Navigate to Reports → Custom Reports\n"
                    "3. Click 'Generate Report' with date range > 90 days\n"
                    "4. Dashboard shows white screen and console logs:\n"
                    "   TypeError: Cannot read properties of undefined (reading 'map')\n\n"
                    "This works fine on Chrome 119. Seems like a Firefox-specific issue.\n\n"
                    "Priority: Medium — affects ~15% of our users on Firefox.\n\n"
                    "Regards,\nDevOps Team, ClientCorp"
                ),
                timestamp="2024-11-15T10:12:00Z",
                has_attachment=False,
                is_reply=False,
            ),
            EmailGroundTruth(
                email_id=_make_id(seed, 2),
                category="technical",
                priority=2,
                department="engineering",
                requires_response=False,
                expected_keywords=[],
            ),
        ),
        (
            EmailData(
                email_id=_make_id(seed, 3),
                sender="mike.johnson@gmail.com",
                sender_name="Mike Johnson",
                subject="Question about your product features",
                body=(
                    "Hello,\n\n"
                    "I came across your product while researching project management "
                    "tools for our small team (8 people). I have a few questions:\n\n"
                    "1. Do you offer a free trial?\n"
                    "2. Can we integrate with Slack and Jira?\n"
                    "3. What's the difference between Pro and Enterprise plans?\n\n"
                    "We're currently evaluating 3-4 options and plan to decide by "
                    "end of month.\n\n"
                    "Best regards,\nMike Johnson"
                ),
                timestamp="2024-11-15T11:30:00Z",
                has_attachment=False,
                is_reply=False,
            ),
            EmailGroundTruth(
                email_id=_make_id(seed, 3),
                category="general",
                priority=4,
                department="support",
                requires_response=False,
                expected_keywords=[],
            ),
        ),
        (
            EmailData(
                email_id=_make_id(seed, 4),
                sender="ops-alert@paymentgateway.com",
                sender_name="Payment Gateway Ops",
                subject="CRITICAL: Payment processing system is DOWN",
                body=(
                    "INCIDENT ALERT — Severity P0\n\n"
                    "All payment processing has halted across production.\n\n"
                    "Impact:\n"
                    "- No transactions are processing since 11:45 UTC\n"
                    "- Estimated ~$50,000/hour in lost revenue\n"
                    "- Customer-facing checkout pages returning 500 errors\n\n"
                    "Root cause investigation ongoing. Database connection pool "
                    "appears exhausted. Need engineering team to scale up "
                    "immediately.\n\n"
                    "— Automated Ops Alert System"
                ),
                timestamp="2024-11-15T11:47:00Z",
                has_attachment=False,
                is_reply=False,
            ),
            EmailGroundTruth(
                email_id=_make_id(seed, 4),
                category="urgent",
                priority=1,
                department="engineering",
                requires_response=False,
                expected_keywords=[],
            ),
        ),
    ]
    return emails


# ═══════════════════════════════════════════════════════════════════════════
#  MEDIUM TASK EMAILS  —  10 emails with ambiguity
# ═══════════════════════════════════════════════════════════════════════════

def _medium_emails(seed: int = 42) -> List[Tuple[EmailData, EmailGroundTruth]]:
    emails = [
        # 1. Billing + urgency (primary = billing)
        (
            EmailData(
                email_id=_make_id(seed, 10),
                sender="angry.customer@enterprise.co",
                sender_name="James Patterson",
                subject="UNAUTHORIZED charges on my account — IMMEDIATE action needed!",
                body=(
                    "I just checked my credit card statement and found THREE "
                    "unauthorized charges from your company:\n"
                    "- $49.99 on Nov 1\n- $49.99 on Nov 5\n- $149.99 on Nov 10\n\n"
                    "I cancelled my subscription on October 15th! This is absolutely "
                    "unacceptable. If these charges are not reversed within 24 hours, "
                    "I will file a dispute with my bank AND report this to the BBB.\n\n"
                    "James Patterson\nAccount: #AC-88123"
                ),
                timestamp="2024-11-15T07:00:00Z",
                has_attachment=False,
                is_reply=False,
            ),
            EmailGroundTruth(
                email_id=_make_id(seed, 10),
                category="billing",
                priority=1,
                department="billing",
                requires_response=True,
                expected_keywords=["refund", "charges", "reversed", "apolog", "cancel"],
            ),
        ),
        # 2. Technical that looks general
        (
            EmailData(
                email_id=_make_id(seed, 11),
                sender="lisa.wong@designstudio.com",
                sender_name="Lisa Wong",
                subject="How do I export my data to CSV?",
                body=(
                    "Hi there,\n\n"
                    "I've been trying to export my project data to CSV but can't "
                    "find the option. I've looked in Settings → Data Management "
                    "and Reports → Export but neither has a CSV option.\n\n"
                    "Am I missing something, or is this feature not available? "
                    "I need the data for a client presentation this Friday.\n\n"
                    "Thanks,\nLisa"
                ),
                timestamp="2024-11-15T08:15:00Z",
                has_attachment=False,
                is_reply=False,
            ),
            EmailGroundTruth(
                email_id=_make_id(seed, 11),
                category="technical",
                priority=3,
                department="support",
                requires_response=False,
                expected_keywords=[],
            ),
        ),
        # 3. Phishing disguised as urgent
        (
            EmailData(
                email_id=_make_id(seed, 12),
                sender="security-alert@acc0unt-verify.net",
                sender_name="Account Security Team",
                subject="⚠️ URGENT: Your account will be SUSPENDED in 12 hours!!!",
                body=(
                    "Dear Valued Customer,\n\n"
                    "We have detected suspicious activity on your account. "
                    "Your account will be permanently suspended unless you verify "
                    "your identity within 12 hours.\n\n"
                    "Click here to verify: http://acc0unt-verify.net/secure-login\n\n"
                    "If you do not verify, all your data will be permanently deleted.\n\n"
                    "Security Team\n"
                    "© 2024 Trusted Company Inc."
                ),
                timestamp="2024-11-15T08:30:00Z",
                has_attachment=False,
                is_reply=False,
            ),
            EmailGroundTruth(
                email_id=_make_id(seed, 12),
                category="spam",
                priority=5,
                department="support",
                requires_response=False,
                expected_keywords=[],
            ),
        ),
        # 4. Technical + billing crossover
        (
            EmailData(
                email_id=_make_id(seed, 13),
                sender="integration-team@megacorp.com",
                sender_name="MegaCorp Integration Team",
                subject="API returning 500 errors — our billing integration is broken",
                body=(
                    "Our automated billing system relies on your REST API "
                    "(endpoint: /api/v2/invoices). Since yesterday 14:00 UTC, "
                    "all requests return HTTP 500 with body:\n"
                    '{"error": "internal_server_error", "trace": "db_timeout"}\n\n'
                    "This is blocking us from processing ~2,000 invoices daily. "
                    "Our finance team cannot reconcile accounts until this is fixed.\n\n"
                    "We need an ETA for resolution ASAP.\n\n"
                    "— MegaCorp Integration Team"
                ),
                timestamp="2024-11-15T09:00:00Z",
                has_attachment=False,
                is_reply=False,
            ),
            EmailGroundTruth(
                email_id=_make_id(seed, 13),
                category="technical",
                priority=1,
                department="engineering",
                requires_response=True,
                expected_keywords=["investigating", "API", "fix", "ETA", "priority"],
            ),
        ),
        # 5. Feature request (general)
        (
            EmailData(
                email_id=_make_id(seed, 14),
                sender="creative.director@agency.com",
                sender_name="Alex Rivera",
                subject="Feature Request: Dark mode support",
                body=(
                    "Hi Product Team,\n\n"
                    "Our design team uses your platform daily, often late into the "
                    "evening. A dark mode option would be incredibly helpful for "
                    "reducing eye strain.\n\n"
                    "Several competitors already offer this. I think it would be "
                    "a great addition to your next release.\n\n"
                    "Happy to provide more detailed feedback if helpful.\n\n"
                    "Best,\nAlex Rivera"
                ),
                timestamp="2024-11-15T09:30:00Z",
                has_attachment=False,
                is_reply=False,
            ),
            EmailGroundTruth(
                email_id=_make_id(seed, 14),
                category="general",
                priority=4,
                department="engineering",
                requires_response=False,
                expected_keywords=[],
            ),
        ),
        # 6. Multi-issue complaint (billing primary)
        (
            EmailData(
                email_id=_make_id(seed, 15),
                sender="frustrated.user@company.org",
                sender_name="Rachel Kim",
                subject="Overcharged AND support chat is broken — two issues",
                body=(
                    "I have TWO problems:\n\n"
                    "1. BILLING: My account was charged $79 for a plan I downgraded "
                    "from last month. I should only be paying $29/month now.\n\n"
                    "2. SUPPORT: When I try to use your live chat to resolve this, "
                    "it just shows 'connecting...' forever. I've tried Chrome, "
                    "Safari, and Firefox. None work.\n\n"
                    "I need the billing issue resolved first — please issue a "
                    "credit for the $50 difference.\n\n"
                    "Rachel Kim"
                ),
                timestamp="2024-11-15T10:00:00Z",
                has_attachment=False,
                is_reply=False,
            ),
            EmailGroundTruth(
                email_id=_make_id(seed, 15),
                category="billing",
                priority=2,
                department="billing",
                requires_response=False,
                expected_keywords=[],
            ),
        ),
        # 7. Urgent escalation
        (
            EmailData(
                email_id=_make_id(seed, 16),
                sender="legal-notices@clientfirm.com",
                sender_name="ClientFirm Legal",
                subject="FWD: Customer threatening legal action over data breach",
                body=(
                    "---------- Forwarded message ----------\n"
                    "From: VP of Operations\n\n"
                    "Team — a customer (Acme Industries, contract value $500K/yr) "
                    "claims their user data was exposed through our API. Their "
                    "legal counsel has sent a formal notice demanding:\n\n"
                    "1. Full incident report within 48 hours\n"
                    "2. Evidence of data protection measures\n"
                    "3. Compensation plan\n\n"
                    "This needs to go to management immediately. Loop in the "
                    "security team as well.\n\n"
                    "— VP Operations"
                ),
                timestamp="2024-11-15T10:30:00Z",
                has_attachment=True,
                is_reply=False,
            ),
            EmailGroundTruth(
                email_id=_make_id(seed, 16),
                category="urgent",
                priority=1,
                department="management",
                requires_response=True,
                expected_keywords=["legal", "security", "incident", "investigate", "priority"],
            ),
        ),
        # 8. Technical docs question
        (
            EmailData(
                email_id=_make_id(seed, 17),
                sender="new.developer@startup.dev",
                sender_name="Carlos Mendez",
                subject="Can't find docs for v3 API migration",
                body=(
                    "Hey,\n\n"
                    "We're trying to migrate from your API v2 to v3 but the "
                    "migration guide linked in the changelog returns a 404.\n\n"
                    "Could you point us to the correct documentation? We have "
                    "a migration deadline of Dec 1st and need to start ASAP.\n\n"
                    "Carlos Mendez\nBackend Engineer"
                ),
                timestamp="2024-11-15T11:00:00Z",
                has_attachment=False,
                is_reply=False,
            ),
            EmailGroundTruth(
                email_id=_make_id(seed, 17),
                category="technical",
                priority=3,
                department="engineering",
                requires_response=False,
                expected_keywords=[],
            ),
        ),
        # 9. Low-priority billing receipt
        (
            EmailData(
                email_id=_make_id(seed, 18),
                sender="receipts@company.com",
                sender_name="Automated Receipts",
                subject="Payment Received — Receipt for Order #ORD-5501",
                body=(
                    "This is an automated confirmation.\n\n"
                    "Payment of $29.00 has been received for your monthly "
                    "subscription (Standard Plan).\n\n"
                    "Invoice: #INV-2024-1102\n"
                    "Date: November 15, 2024\n"
                    "Amount: $29.00\n"
                    "Method: Visa ending in 4242\n\n"
                    "No action required. Thank you for being a customer!"
                ),
                timestamp="2024-11-15T11:15:00Z",
                has_attachment=True,
                is_reply=False,
            ),
            EmailGroundTruth(
                email_id=_make_id(seed, 18),
                category="billing",
                priority=5,
                department="billing",
                requires_response=False,
                expected_keywords=[],
            ),
        ),
        # 10. Multi-language support
        (
            EmailData(
                email_id=_make_id(seed, 19),
                sender="usuario@correo.mx",
                sender_name="María García",
                subject="Necesito ayuda con mi cuenta / I need help with my account",
                body=(
                    "Hola / Hello,\n\n"
                    "No puedo acceder a mi cuenta desde hace 3 días. Recibo un "
                    "mensaje de error que dice 'Account locked'. Ya intenté "
                    "restablecer mi contraseña pero el enlace no funciona.\n\n"
                    "I cannot access my account for 3 days. I get an error "
                    "'Account locked'. I already tried resetting my password "
                    "but the link doesn't work.\n\n"
                    "Por favor ayúdenme / Please help me.\n\n"
                    "María García\nAccount: #AC-55209"
                ),
                timestamp="2024-11-15T11:30:00Z",
                has_attachment=False,
                is_reply=False,
            ),
            EmailGroundTruth(
                email_id=_make_id(seed, 19),
                category="technical",
                priority=2,
                department="support",
                requires_response=False,
                expected_keywords=[],
            ),
        ),
    ]
    return emails


# ═══════════════════════════════════════════════════════════════════════════
#  HARD TASK EMAILS  —  20 emails with threads, ambiguity, red-herrings
# ═══════════════════════════════════════════════════════════════════════════

def _hard_emails(seed: int = 42) -> List[Tuple[EmailData, EmailGroundTruth]]:
    # Start with the medium emails (10) and add 10 more complex ones
    base = _medium_emails(seed)
    thread_a = f"thread-{_make_id(seed, 100)}"
    thread_b = f"thread-{_make_id(seed, 200)}"

    extra = [
        # 11. Thread A — first message (customer complaint)
        (
            EmailData(
                email_id=_make_id(seed, 20),
                sender="vip.client@fortune500.com",
                sender_name="David Chen, VP Engineering",
                subject="Performance degradation in production — 5x slower",
                body=(
                    "Our production workloads using your service have become "
                    "5x slower since your last update (v4.2.1 released Nov 12).\n\n"
                    "Metrics:\n"
                    "- Average response time: 250ms → 1,200ms\n"
                    "- P99 latency: 500ms → 5,000ms\n"
                    "- Throughput dropped from 10K to 2K req/s\n\n"
                    "This is impacting our customer-facing products. We're an "
                    "Enterprise customer paying $250K/year. Need this fixed TODAY.\n\n"
                    "David Chen\nVP Engineering, Fortune500 Corp"
                ),
                timestamp="2024-11-15T06:00:00Z",
                has_attachment=False,
                is_reply=False,
                thread_id=thread_a,
            ),
            EmailGroundTruth(
                email_id=_make_id(seed, 20),
                category="technical",
                priority=1,
                department="engineering",
                requires_response=True,
                expected_keywords=["investigating", "performance", "update", "rollback", "priority"],
            ),
        ),
        # 12. Thread A — follow-up (escalation)
        (
            EmailData(
                email_id=_make_id(seed, 21),
                sender="vip.client@fortune500.com",
                sender_name="David Chen, VP Engineering",
                subject="Re: Performance degradation in production — 5x slower",
                body=(
                    "It's been 4 hours with no response. I'm looping in your "
                    "CEO's office. If this isn't resolved by EOD, we will begin "
                    "evaluating alternative providers.\n\n"
                    "Our CTO is requesting a call with your engineering lead.\n\n"
                    "David Chen"
                ),
                timestamp="2024-11-15T10:00:00Z",
                has_attachment=False,
                is_reply=True,
                thread_id=thread_a,
            ),
            EmailGroundTruth(
                email_id=_make_id(seed, 21),
                category="urgent",
                priority=1,
                department="management",
                requires_response=True,
                expected_keywords=["apolog", "escalat", "call", "engineering", "resolv"],
            ),
        ),
        # 13. Red herring — looks urgent but is marketing
        (
            EmailData(
                email_id=_make_id(seed, 22),
                sender="deals@saas-marketplace.com",
                sender_name="SaaS Marketplace",
                subject="⚡ URGENT: Black Friday Deal Expires in 2 Hours!",
                body=(
                    "Don't miss out! Our biggest sale of the year ends TONIGHT.\n\n"
                    "🔥 50% OFF all annual plans\n"
                    "🔥 Free migration from competitor products\n"
                    "🔥 Bonus: 3 months free support\n\n"
                    "Use code BLACKFRIDAY50 at checkout.\n\n"
                    "This email was sent to subscribers of SaaS Marketplace.\n"
                    "Unsubscribe: http://saas-marketplace.com/unsub"
                ),
                timestamp="2024-11-15T09:00:00Z",
                has_attachment=False,
                is_reply=False,
            ),
            EmailGroundTruth(
                email_id=_make_id(seed, 22),
                category="spam",
                priority=5,
                department="support",
                requires_response=False,
                expected_keywords=[],
            ),
        ),
        # 14. Thread B — Internal IT request
        (
            EmailData(
                email_id=_make_id(seed, 23),
                sender="hr@company-internal.com",
                sender_name="HR Department",
                subject="New employee onboarding — system access needed",
                body=(
                    "Hi IT Team,\n\n"
                    "We have 3 new engineers starting Monday (Nov 18):\n"
                    "- Priya Sharma (priya.s@company.com)\n"
                    "- Tom Williams (tom.w@company.com)\n"
                    "- Yuki Tanaka (yuki.t@company.com)\n\n"
                    "Please set up the following for each:\n"
                    "- GitHub organization access\n"
                    "- Slack workspace invitation\n"
                    "- VPN credentials\n"
                    "- Development environment provisioning\n\n"
                    "Deadline: Friday Nov 15 EOD.\n\n"
                    "Thanks,\nHR Department"
                ),
                timestamp="2024-11-15T07:30:00Z",
                has_attachment=True,
                is_reply=False,
                thread_id=thread_b,
            ),
            EmailGroundTruth(
                email_id=_make_id(seed, 23),
                category="general",
                priority=2,
                department="engineering",
                requires_response=False,
                expected_keywords=[],
            ),
        ),
        # 15. Thread B — follow-up
        (
            EmailData(
                email_id=_make_id(seed, 24),
                sender="hr@company-internal.com",
                sender_name="HR Department",
                subject="Re: New employee onboarding — system access needed",
                body=(
                    "Quick update — Yuki Tanaka's start date has been moved to "
                    "Nov 25. Only Priya and Tom need access by this Friday.\n\n"
                    "Also, Priya needs admin access to the staging environment "
                    "as she'll be leading the QA team.\n\n"
                    "Thanks!"
                ),
                timestamp="2024-11-15T09:45:00Z",
                has_attachment=False,
                is_reply=True,
                thread_id=thread_b,
            ),
            EmailGroundTruth(
                email_id=_make_id(seed, 24),
                category="general",
                priority=3,
                department="engineering",
                requires_response=False,
                expected_keywords=[],
            ),
        ),
        # 16. Red herring #2 — sounds critical but is a test
        (
            EmailData(
                email_id=_make_id(seed, 25),
                sender="qa-automation@company-internal.com",
                sender_name="QA Automation Bot",
                subject="[TEST ENV] ALERT: Database corruption detected!",
                body=(
                    "⚠️ DATABASE CORRUPTION DETECTED ⚠️\n\n"
                    "Environment: STAGING (test-db-replica-03)\n"
                    "Severity: CRITICAL\n\n"
                    "Note: This is a SCHEDULED TEST of our alerting system. "
                    "No actual corruption has occurred. This alert was triggered "
                    "as part of our quarterly disaster recovery drill.\n\n"
                    "If you received this in error, please contact the QA team.\n\n"
                    "— QA Automation"
                ),
                timestamp="2024-11-15T08:00:00Z",
                has_attachment=False,
                is_reply=False,
            ),
            EmailGroundTruth(
                email_id=_make_id(seed, 25),
                category="general",
                priority=4,
                department="engineering",
                requires_response=False,
                expected_keywords=[],
            ),
        ),
        # 17. Complex technical with compliance implications
        (
            EmailData(
                email_id=_make_id(seed, 26),
                sender="compliance@eu-regulator.gov",
                sender_name="EU Data Protection Authority",
                subject="GDPR Audit Notice — Response Required Within 30 Days",
                body=(
                    "Dear Data Controller,\n\n"
                    "As part of our routine compliance program, your organization "
                    "has been selected for a GDPR data processing audit.\n\n"
                    "Please provide the following within 30 calendar days:\n"
                    "1. Complete data processing inventory\n"
                    "2. Records of consent management\n"
                    "3. Data breach notification procedures\n"
                    "4. Third-party processor agreements\n\n"
                    "Failure to comply may result in penalties up to 4% of "
                    "annual global turnover.\n\n"
                    "Reference: AUDIT-2024-EU-7819\n\n"
                    "EU Data Protection Authority"
                ),
                timestamp="2024-11-15T06:30:00Z",
                has_attachment=True,
                is_reply=False,
            ),
            EmailGroundTruth(
                email_id=_make_id(seed, 26),
                category="urgent",
                priority=1,
                department="management",
                requires_response=True,
                expected_keywords=["compliance", "GDPR", "audit", "legal", "respond"],
            ),
        ),
        # 18. Subtle spam — newsletter opt-out
        (
            EmailData(
                email_id=_make_id(seed, 27),
                sender="newsletter@tech-digest-daily.com",
                sender_name="Tech Digest Daily",
                subject="Your Weekly AI Roundup — Issue #247",
                body=(
                    "This week in AI:\n\n"
                    "• OpenAI launches GPT-5 with multimodal reasoning\n"
                    "• Google DeepMind achieves AGI benchmark (claimed)\n"
                    "• Meta releases Llama 4 — biggest open-source model yet\n\n"
                    "Read full articles: http://tech-digest-daily.com/issue247\n\n"
                    "You're receiving this because you signed up at TechConf 2023.\n"
                    "Unsubscribe: http://tech-digest-daily.com/unsub?id=x8291"
                ),
                timestamp="2024-11-15T05:00:00Z",
                has_attachment=False,
                is_reply=False,
            ),
            EmailGroundTruth(
                email_id=_make_id(seed, 27),
                category="spam",
                priority=5,
                department="support",
                requires_response=False,
                expected_keywords=[],
            ),
        ),
        # 19. Customer success — upsell opportunity (general)
        (
            EmailData(
                email_id=_make_id(seed, 28),
                sender="partnership@bigcompany.com",
                sender_name="BigCompany Partnerships",
                subject="Partnership Proposal — Joint Product Integration",
                body=(
                    "Hi Team,\n\n"
                    "I'm reaching out from BigCompany's partnership team. We've "
                    "been using your product internally and see a great opportunity "
                    "for a formal partnership:\n\n"
                    "- Co-marketing arrangement\n"
                    "- Native integration between our platforms\n"
                    "- Joint webinar series for Q1 2025\n\n"
                    "We have 50K+ enterprise users who could benefit. Would love "
                    "to schedule a call next week to discuss.\n\n"
                    "Best,\nPartnership Team, BigCompany"
                ),
                timestamp="2024-11-15T10:15:00Z",
                has_attachment=True,
                is_reply=False,
            ),
            EmailGroundTruth(
                email_id=_make_id(seed, 28),
                category="general",
                priority=3,
                department="management",
                requires_response=False,
                expected_keywords=[],
            ),
        ),
        # 20. Technical security vulnerability report
        (
            EmailData(
                email_id=_make_id(seed, 29),
                sender="security-researcher@bugbounty.io",
                sender_name="Ethical Hacker — Bug Bounty",
                subject="Responsible Disclosure: XSS vulnerability in user profile page",
                body=(
                    "Hello Security Team,\n\n"
                    "I've discovered a stored XSS vulnerability in your user "
                    "profile page. An attacker can inject malicious JavaScript "
                    "via the 'bio' field which executes when other users view "
                    "the profile.\n\n"
                    "Steps to reproduce:\n"
                    "1. Edit profile → Bio field\n"
                    '2. Enter: <script>alert(document.cookie)</script>\n'
                    "3. Save and view profile from another account\n"
                    "4. Cookie data is exposed\n\n"
                    "I've followed responsible disclosure practices and will "
                    "grant 90 days before public disclosure.\n\n"
                    "CVSS Score: 7.1 (High)\n\n"
                    "— Security Researcher, Bug Bounty Platform"
                ),
                timestamp="2024-11-15T11:45:00Z",
                has_attachment=False,
                is_reply=False,
            ),
            EmailGroundTruth(
                email_id=_make_id(seed, 29),
                category="urgent",
                priority=1,
                department="engineering",
                requires_response=True,
                expected_keywords=["security", "XSS", "fix", "patch", "acknowledge", "thank"],
            ),
        ),
    ]
    return base + extra


# ═══════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════

def generate_emails(
    task_id: str, seed: int = 42
) -> Tuple[List[EmailData], List[EmailGroundTruth]]:
    """Generate emails and ground truth for the specified task.

    Args:
        task_id: One of 'easy', 'medium', 'hard'
        seed: Random seed for reproducibility

    Returns:
        Tuple of (emails, ground_truths)
    """
    generators = {
        "easy": _easy_emails,
        "medium": _medium_emails,
        "hard": _hard_emails,
    }

    gen_fn = generators.get(task_id)
    if gen_fn is None:
        raise ValueError(f"Unknown task_id: {task_id}. Must be one of {list(generators.keys())}")

    pairs = gen_fn(seed)
    emails = [e for e, _ in pairs]
    truths = [t for _, t in pairs]
    return emails, truths
