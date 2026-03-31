"""
Task: Easy — FAQ / Password Reset
Goal: Agent searches KB and sends a correct, helpful reply. Closes the ticket in ≤5 steps.
Reward signals: correct tool use, helpful answer, efficient resolution.
"""

import json
import pathlib
import random

_DATA = pathlib.Path(__file__).parent.parent / "data" / "ticket_templates.json"
with open(_DATA) as f:
    _TEMPLATES = json.load(f)["easy"]


AVAILABLE_TOOLS = [
    "search_kb",
    "ask_clarification",
    "send_reply",
    "update_ticket",
    "close_ticket",
]


def make_state(seed: int = None) -> dict:
    rng = random.Random(seed)
    template = rng.choice(_TEMPLATES)

    return {
        # Core ticket
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
        # Conversation so far
        "conversation": [
            {
                "role": "customer",
                "content": template["description"],
            }
        ],
        # Task metadata
        "task_id": "easy",
        "ticket_template": template,
        "step_count": 0,
        "max_steps": 5,
        "tools_used": [],
        "done": False,
    }
