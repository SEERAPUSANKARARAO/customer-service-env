"""
Task: Hard — Angry Refund with Escalation Decision
Goal: De-escalate, verify order, issue correct refund, decide on escalation appropriately.
Tests: empathy, multi-tool orchestration, policy adherence, escalation judgment.
"""

import json
import pathlib
import random

_DATA = pathlib.Path(__file__).parent.parent / "data" / "ticket_templates.json"
with open(_DATA) as f:
    _TEMPLATES = json.load(f)["hard"]


AVAILABLE_TOOLS = [
    "search_kb",
    "ask_clarification",
    "get_order_details",
    "send_reply",
    "update_ticket",
    "issue_refund",
    "escalate_to_human",
    "close_ticket",
]


def make_state(seed: int = None) -> dict:
    rng = random.Random(seed)
    template = rng.choice(_TEMPLATES)

    return {
        "ticket": {
            "id": template["id"],
            "subject": template["subject"],
            "description": template["description"],
            "status": "open",
            "customer_info": template["customer_info"],
            "order_id": template.get("order_id"),
            "sentiment": template.get("sentiment", "angry"),
            "notes": [],
            "resolved": False,
            "escalated": False,
            "refund_issued": 0.0,
            "refund_reason": None,
            "escalation_reason": None,
        },
        "conversation": [
            {
                "role": "customer",
                "content": template["description"],
            }
        ],
        "task_id": "hard",
        "ticket_template": template,
        # Ground truth for grader
        "correct_refund_amount": template.get("correct_refund_amount", 79.99),
        "should_escalate": template.get("should_escalate", False),
        "step_count": 0,
        "max_steps": 10,
        "tools_used": [],
        "done": False,
    }
