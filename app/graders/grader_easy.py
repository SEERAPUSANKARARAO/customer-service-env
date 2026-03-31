"""
Grader: Easy Task — FAQ / Password Reset
Scoring breakdown (max 1.0):
  +0.20  agent used search_kb
  +0.30  agent reply contains relevant KB answer content
  +0.30  ticket closed / resolved
  +0.20  efficiency bonus: resolved in ≤4 steps
  -0.30  penalty: unnecessary escalation
  -0.10  penalty: rude or unhelpful reply
"""

# Keywords per KB article — expanded to catch paraphrased answers
_RESOLUTION_KEYWORDS = {
    "password_reset": [
        "forgot password", "reset link", "password reset", "30 minutes",
        "spam", "junk", "forgot", "login page", "click", "email address",
    ],
    "order_cancellation": [
        "cancel", "1 hour", "full refund", "dispatch", "cancelled",
        "within 1 hour", "not yet dispatched", "cancellation",
    ],
    "payment_methods": [
        "upi", "visa", "mastercard", "paypal", "credit card",
        "digital wallet", "american express", "ssl", "debit",
    ],
    "shipping_policy": [
        "5-7", "business days", "track", "express", "standard delivery",
        "1-2 business", "international", "dispatch",
    ],
    "warranty": [
        "warranty", "1-year", "manufacturer", "defect", "pickup",
        "replacement", "7 business", "photos", "video",
    ],
    "account_deletion": [
        "delete", "settings", "privacy", "gdpr", "permanent",
        "30 days", "deactivate", "erase",
    ],
    "duplicate_charge": [
        "pending", "24-48 hours", "auto-resolve", "bank statement",
        "screenshot", "investigate", "3 business days", "duplicate",
        "48 hours", "charge", "refund",
    ],
    "refund_policy": [
        "refund", "30 days", "defective", "3-5 business", "return",
        "change-of-mind", "partial",
    ],
    "product_availability": [
        "restock", "7-14 days", "notify me", "out of stock", "warehouse",
    ],
    "contact_support": [
        "live chat", "support@", "phone", "9am", "1800", "email",
    ],
}

_RUDE_KEYWORDS = [
    "that's not possible", "you should know", "read the faq",
    "obviously", "as stated", "already told you",
]


def grade(state: dict) -> dict:
    """Returns dict with 'score' (0.0-1.0) and 'breakdown'."""
    ticket     = state["ticket"]
    tools_used = state.get("tools_used", [])
    conv       = state.get("conversation", [])
    step_count = state.get("step_count", 0)
    template   = state.get("ticket_template", {})

    breakdown = {}
    score     = 0.0

    agent_text = " ".join(
        m["content"].lower() for m in conv if m.get("role") == "agent"
    )

    # +0.20: used search_kb
    if "search_kb" in tools_used:
        breakdown["used_search_kb"] = 0.20
        score += 0.20
    else:
        breakdown["used_search_kb"] = 0.0

    # +0.30: reply contains relevant KB answer content
    expected_kb_key = template.get("expected_kb_key", "")
    relevant_kws    = _RESOLUTION_KEYWORDS.get(expected_kb_key, [])

    if relevant_kws and any(kw in agent_text for kw in relevant_kws):
        breakdown["relevant_answer_given"] = 0.30
        score += 0.30
    elif any(len(m["content"]) > 60 for m in conv if m.get("role") == "agent"):
        # Gave a substantial reply even if exact keywords missed — partial credit
        breakdown["relevant_answer_given"] = 0.15
        score += 0.15
    else:
        breakdown["relevant_answer_given"] = 0.0

    # +0.30: ticket closed / resolved
    if ticket.get("resolved") or ticket.get("status") == "closed":
        breakdown["ticket_resolved"] = 0.30
        score += 0.30
    else:
        breakdown["ticket_resolved"] = 0.0

    # +0.20: efficiency bonus — resolved in ≤4 steps
    if step_count <= 4 and (ticket.get("resolved") or ticket.get("status") == "closed"):
        breakdown["efficiency_bonus"] = 0.20
        score += 0.20
    else:
        breakdown["efficiency_bonus"] = 0.0

    # -0.30: unnecessary escalation (simple FAQ should never escalate)
    if ticket.get("escalated"):
        breakdown["unnecessary_escalation_penalty"] = -0.30
        score = max(0.0, score - 0.30)
    else:
        breakdown["unnecessary_escalation_penalty"] = 0.0

    # -0.10: rude reply
    if any(kw in agent_text for kw in _RUDE_KEYWORDS):
        breakdown["rude_reply_penalty"] = -0.10
        score = max(0.0, score - 0.10)
    else:
        breakdown["rude_reply_penalty"] = 0.0

    final_score = round(min(max(score, 0.0), 1.0), 4)
    return {"score": final_score, "breakdown": breakdown}
