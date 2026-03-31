"""
Grader: Medium Task — Missing Order Investigation
Scoring breakdown (max 1.0):
  +0.15  asked a clarifying question (for order ID)
  +0.20  called get_order_details successfully
  +0.25  communicated correct order status to customer
  +0.25  ticket closed/resolved
  +0.15  efficiency bonus: resolved in ≤5 steps
  -0.15  penalty: gave status info without checking order first
  -0.15  penalty: issued refund when order is still in_transit (premature)
  -0.10  penalty: unnecessary escalation
  -0.10  penalty: rude reply
"""

_STATUS_KEYWORDS = {
    "in_transit": ["in transit", "on its way", "out for delivery", "estimated",
                   "tracking", "shipping", "stalled", "delayed"],
    "delivered":  ["delivered", "already delivered", "delivery date", "was delivered"],
    "lost":       ["lost", "cannot locate", "missing", "full refund", "lost in transit"],
    "delayed":    ["delayed", "longer than expected", "apologies for the delay"],
}

_RUDE_KEYWORDS = [
    "that's not my problem", "your fault", "should have", "obviously", "as i said",
]


def grade(state: dict) -> dict:
    ticket     = state["ticket"]
    tools_used = state.get("tools_used", [])
    conv       = state.get("conversation", [])
    step_count = state.get("step_count", 0)

    breakdown = {}
    score     = 0.0

    agent_text = " ".join(
        m["content"].lower() for m in conv if m.get("role") == "agent"
    )

    # Determine actual order status from template
    template      = state.get("ticket_template", {})
    from app.tools import ORDER_DATABASE
    order_id      = template.get("order_to_find", "ORD-1002")
    actual_status = ORDER_DATABASE.get(order_id, {}).get("status", "unknown")
    status_kws    = _STATUS_KEYWORDS.get(actual_status, [])

    # +0.15: asked clarifying question
    if "ask_clarification" in tools_used:
        breakdown["asked_clarification"] = 0.15
        score += 0.15
    else:
        breakdown["asked_clarification"] = 0.0

    # +0.20: fetched order details
    if "get_order_details" in tools_used:
        breakdown["fetched_order_details"] = 0.20
        score += 0.20
    else:
        breakdown["fetched_order_details"] = 0.0

    # +0.25: communicated correct status
    if status_kws and any(kw in agent_text for kw in status_kws):
        breakdown["correct_status_communicated"] = 0.25
        score += 0.25
    elif any(len(m["content"]) > 60 for m in conv if m.get("role") == "agent"):
        breakdown["correct_status_communicated"] = 0.10
        score += 0.10
    else:
        breakdown["correct_status_communicated"] = 0.0

    # +0.25: ticket closed/resolved
    if ticket.get("resolved") or ticket.get("status") == "closed":
        breakdown["ticket_resolved"] = 0.25
        score += 0.25
    else:
        breakdown["ticket_resolved"] = 0.0

    # +0.15: efficiency bonus ≤5 steps
    if step_count <= 5 and (ticket.get("resolved") or ticket.get("status") == "closed"):
        breakdown["efficiency_bonus"] = 0.15
        score += 0.15
    else:
        breakdown["efficiency_bonus"] = 0.0

    # -0.15: gave status info without checking order first
    if "get_order_details" not in tools_used and any(
        kw in agent_text for kw in ["in transit", "delivered", "lost", "tracking"]
    ):
        breakdown["info_without_checking_penalty"] = -0.15
        score = max(0.0, score - 0.15)
    else:
        breakdown["info_without_checking_penalty"] = 0.0

    # -0.15: issued refund when order is just in_transit (premature — should wait)
    if "issue_refund" in tools_used and actual_status == "in_transit":
        breakdown["premature_refund_penalty"] = -0.15
        score = max(0.0, score - 0.15)
    else:
        breakdown["premature_refund_penalty"] = 0.0

    # -0.10: unnecessary escalation
    if ticket.get("escalated") and actual_status != "lost":
        breakdown["unnecessary_escalation_penalty"] = -0.10
        score = max(0.0, score - 0.10)
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
