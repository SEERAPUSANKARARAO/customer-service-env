"""
Task: Medium — Missing Order Investigation
Goal: Agent must ask for order ID, look up order status, communicate it clearly, and close ticket.
Tests: multi-turn reasoning, clarifying questions, tool orchestration.
"""

import json
import pathlib
import random

_DATA = pathlib.Path(__file__).parent.parent / "data" / "ticket_templates.json"
with open(_DATA) as f:
    _TEMPLATES = json.load(f)["medium"]


AVAILABLE_TOOLS = [
    "search_kb",
    "ask_clarification",
    "get_order_details",
    "send_reply",
    "update_ticket",
    "issue_refund",
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
        "task_id": "medium",
        "ticket_template": template,
        "step_count": 0,
        "max_steps": 8,
        "tools_used": [],
        "done": False,
    }
