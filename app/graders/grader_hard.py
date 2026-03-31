"""
Grader: Hard Task — Angry Refund with Escalation Decision
Scoring breakdown (max 1.0):
  +0.10  fetched order details (verified before acting)
  +0.10  used send_reply at least once
  +0.20  empathetic language detected in replies
  +0.20  refund issued (any amount)
  +0.15  correct refund amount (within 10% tolerance)
  +0.15  ticket closed/resolved
  +0.10  correct escalation decision (escalated when should, didn't when shouldn't)
  -0.25  wrong escalation decision
  -0.15  rude or dismissive language (2+ occurrences)
  -0.10  issued refund without verifying order first
  -0.10  refused refund on a valid claim
"""

_EMPATHY_KEYWORDS = [
    "understand", "sorry", "apologize", "apologies", "frustrat",
    "absolutely", "assure", "concern", "inconvenience", "sincerely",
    "i hear you", "i can see", "that must be", "completely understand",
    "regret", "deeply sorry",
]

_RUDE_KEYWORDS = [
    "your fault", "you should have", "that's impossible",
    "not my problem", "as i said", "already told you",
    "clearly", "obviously", "just follow",
]

_DENIAL_KEYWORDS = [
    "cannot refund", "can't refund", "not eligible", "no refund",
    "policy doesn't allow", "denied", "we don't offer",
]


def grade(state: dict) -> dict:
    ticket = state["ticket"]
    tools_used = state.get("tools_used", [])
    conversation = state.get("conversation", [])

    agent_messages = [m for m in conversation if m.get("role") == "agent"]
    agent_text = " ".join(m["content"].lower() for m in agent_messages)

    # Ground truth from task state
    correct_refund     = state.get("correct_refund_amount", 79.99)
    should_escalate    = state.get("should_escalate", False)
    actually_escalated = ticket.get("escalated", False)
    refund_issued      = ticket.get("refund_issued", 0.0)

    breakdown = {}
    score     = 0.0

    # +0.10: verified order before acting
    if "get_order_details" in tools_used:
        breakdown["verified_order"] = 0.10
        score += 0.10
    else:
        breakdown["verified_order"] = 0.0

    # +0.10: sent at least one reply
    if "send_reply" in tools_used or any(m.get("role") == "agent" for m in conversation):
        breakdown["sent_reply"] = 0.10
        score += 0.10
    else:
        breakdown["sent_reply"] = 0.0

    # +0.20: empathetic language
    empathy_hits = sum(1 for kw in _EMPATHY_KEYWORDS if kw in agent_text)
    if empathy_hits >= 3:
        breakdown["empathy_score"] = 0.20
        score += 0.20
    elif empathy_hits >= 1:
        breakdown["empathy_score"] = 0.10
        score += 0.10
    else:
        breakdown["empathy_score"] = 0.0

    # +0.20: issued a refund
    if "issue_refund" in tools_used and refund_issued > 0:
        breakdown["refund_issued"] = 0.20
        score += 0.20
    else:
        breakdown["refund_issued"] = 0.0

    # +0.15: correct refund amount
    # Accept: exact item price, double item price ("charged twice"),
    # or any amount within 10% of either.
    if refund_issued > 0:
        double_refund    = correct_refund * 2
        tight_tolerance  = correct_refund * 0.10

        exact_match  = abs(refund_issued - correct_refund) <= tight_tolerance
        double_match = abs(refund_issued - double_refund)  <= tight_tolerance

        if exact_match or double_match:
            breakdown["correct_refund_amount"] = 0.15
            score += 0.15
        elif refund_issued >= correct_refund * 0.5:
            # Reasonable partial refund — half credit
            breakdown["correct_refund_amount"] = 0.07
            score += 0.07
        else:
            breakdown["correct_refund_amount"] = 0.0
    else:
        breakdown["correct_refund_amount"] = 0.0

    # +0.15: ticket closed/resolved
    if ticket.get("resolved") or ticket.get("status") in ("closed", "escalated"):
        breakdown["ticket_resolved"] = 0.15
        score += 0.15
    else:
        breakdown["ticket_resolved"] = 0.0

    # +0.10: correct escalation decision
    escalation_correct = (should_escalate == actually_escalated)
    if escalation_correct:
        breakdown["correct_escalation_decision"] = 0.10
        score += 0.10
    else:
        breakdown["correct_escalation_decision"] = 0.0

    # --- Penalties ---

    # -0.25: wrong escalation decision
    if not escalation_correct:
        breakdown["wrong_escalation_penalty"] = -0.25
        score = max(0.0, score - 0.25)
    else:
        breakdown["wrong_escalation_penalty"] = 0.0

    # -0.15: rude or dismissive language
    rude_hits = sum(1 for kw in _RUDE_KEYWORDS if kw in agent_text)
    if rude_hits >= 2:
        breakdown["rude_language_penalty"] = -0.15
        score = max(0.0, score - 0.15)
    elif rude_hits == 1:
        breakdown["rude_language_penalty"] = -0.07
        score = max(0.0, score - 0.07)
    else:
        breakdown["rude_language_penalty"] = 0.0

    # -0.10: issued refund without checking order first
    tools_sequence = tools_used
    if "issue_refund" in tools_sequence and "get_order_details" not in tools_sequence:
        breakdown["refund_without_verification_penalty"] = -0.10
        score = max(0.0, score - 0.10)
    elif ("issue_refund" in tools_sequence and "get_order_details" in tools_sequence and
          tools_sequence.index("issue_refund") < tools_sequence.index("get_order_details")):
        breakdown["refund_without_verification_penalty"] = -0.10
        score = max(0.0, score - 0.10)
    else:
        breakdown["refund_without_verification_penalty"] = 0.0

    # -0.10: denied a valid refund claim
    if any(kw in agent_text for kw in _DENIAL_KEYWORDS) and correct_refund > 0:
        breakdown["unjust_denial_penalty"] = -0.10
        score = max(0.0, score - 0.10)
    else:
        breakdown["unjust_denial_penalty"] = 0.0

    final_score = round(min(max(score, 0.0), 1.0), 4)
    return {"score": final_score, "breakdown": breakdown}
