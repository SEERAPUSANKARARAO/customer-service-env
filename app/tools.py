"""
tools.py
Simulated tool implementations for the Customer Service environment.
All tools operate on the mutable state dict and return (tool_result, updated_state).
"""

import json
import pathlib

# ---------------------------------------------------------------------------
# Static data
# ---------------------------------------------------------------------------

_DATA_DIR = pathlib.Path(__file__).parent / "data"

with open(_DATA_DIR / "knowledge_base.json") as f:
    KNOWLEDGE_BASE = json.load(f)

# Simulated order database
ORDER_DATABASE = {
    "ORD-1001": {
        "order_id": "ORD-1001",
        "status": "delivered",
        "item": "Laptop Stand",
        "amount": 49.99,
        "delivery_date": "2026-03-20",
        "tracking_url": "https://track.example.com/ORD-1001",
    },
    "ORD-1002": {
        "order_id": "ORD-1002",
        "status": "in_transit",
        "item": "USB Hub",
        "amount": 29.99,
        "estimated_delivery": "2026-03-30",
        "tracking_url": "https://track.example.com/ORD-1002",
        "last_location": "Mumbai Sorting Facility",
    },
    "ORD-1003": {
        "order_id": "ORD-1003",
        "status": "lost",
        "item": "Webcam HD Pro",
        "amount": 79.99,
        "notes": "Package marked as lost by courier. Full refund eligible.",
        "eligible_refund": 79.99,
    },
    "ORD-1004": {
        "order_id": "ORD-1004",
        "status": "in_transit",
        "item": "Mechanical Keyboard",
        "amount": 89.99,
        "estimated_delivery": "2026-03-31",
        "tracking_url": "https://track.example.com/ORD-1004",
        "last_location": "Delhi Distribution Centre",
        "tracking_stalled_days": 5,
    },
    "ORD-1005": {
        "order_id": "ORD-1005",
        "status": "lost",
        "item": "Noise Cancelling Headphones",
        "amount": 149.99,
        "notes": "Package lost in transit. Customer contacted 4 times. Premium customer — escalation recommended.",
        "eligible_refund": 149.99,
        "premium_customer": True,
    },
    "ORD-1006": {
        "order_id": "ORD-1006",
        "status": "delivered",
        "item": "Wireless Mouse",
        "amount": 34.99,
        "delivery_date": "2026-03-22",
        "tracking_url": "https://track.example.com/ORD-1006",
        "notes": "Delivered and signed for.",
    },
}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def search_kb(state: dict, query: str) -> tuple:
    """
    Search the knowledge base for a relevant answer.
    Returns (result_dict, state) — state is not modified.
    """
    q = query.lower()
    best_match = None
    best_score = 0

    for key, entry in KNOWLEDGE_BASE.items():
        score = sum(1 for kw in entry["keywords"] if kw in q)
        if score > best_score:
            best_score = score
            best_match = (key, entry)

    if best_match and best_score > 0:
        key, entry = best_match
        result = {
            "found": True,
            "kb_key": key,
            "answer": entry["answer"],
            "category": entry.get("category", "general"),
        }
    else:
        result = {
            "found": False,
            "answer": None,
            "message": "No relevant article found. Consider asking the customer for more details.",
        }

    return result, state


def get_order_details(state: dict, order_id: str) -> tuple:
    """
    Fetch order details from the simulated order database.
    """
    order_id = order_id.strip().upper()
    if order_id in ORDER_DATABASE:
        result = {"found": True, "order": ORDER_DATABASE[order_id]}
    else:
        result = {
            "found": False,
            "error": f"Order '{order_id}' not found. Please verify the order ID with the customer.",
        }
    return result, state


def update_ticket(state: dict, status: str, note: str) -> tuple:
    """
    Update the ticket status and add a note.
    """
    valid_statuses = ["open", "in_progress", "pending_customer", "resolved", "closed", "escalated"]
    if status not in valid_statuses:
        return {"error": f"Invalid status '{status}'. Choose from: {valid_statuses}"}, state

    state["ticket"]["status"] = status
    state["ticket"]["notes"].append(note)
    result = {"success": True, "new_status": status, "note_added": note}
    return result, state


def send_reply(state: dict, message: str, tone: str = "professional") -> tuple:
    """
    Send a reply message to the customer.
    tone: "professional" | "empathetic" | "apologetic" | "formal"
    """
    valid_tones = ["professional", "empathetic", "apologetic", "formal", "friendly"]
    if tone not in valid_tones:
        tone = "professional"

    state["conversation"].append({
        "role": "agent",
        "content": message,
        "tone": tone,
    })
    result = {"success": True, "message_sent": message, "tone": tone}
    return result, state


def ask_clarification(state: dict, question: str) -> tuple:
    """
    Ask the customer a clarifying question.
    Simulates the customer responding with relevant info (order ID, etc.).
    """
    state["conversation"].append({
        "role": "agent",
        "content": question,
        "tone": "professional",
    })

    # Simulate customer response based on question content
    q_lower = question.lower()
    task_id = state.get("task_id", "")

    if any(kw in q_lower for kw in ["order id", "order number", "reference number"]):
        # Give the customer's order ID based on task
        if task_id == "medium":
            templates = state.get("ticket_template", {})
            order_id = templates.get("order_to_find", "ORD-1002")
            customer_reply = f"Sure! My order ID is {order_id}."
        elif task_id == "hard":
            order_id = state["ticket"].get("order_id", "ORD-1003")
            customer_reply = f"It's {order_id}. Please sort this out immediately!"
        else:
            customer_reply = "My order ID is ORD-1001."
    elif any(kw in q_lower for kw in ["email", "account email", "registered"]):
        customer_reply = f"My email is {state['ticket']['customer_info']['email']}."
    elif any(kw in q_lower for kw in ["describe", "more detail", "what happened", "explain"]):
        customer_reply = "I've already explained everything in my original message."
    else:
        customer_reply = "Yes, that's correct."

    state["conversation"].append({
        "role": "customer",
        "content": customer_reply,
    })

    result = {
        "success": True,
        "question_asked": question,
        "customer_reply": customer_reply,
    }
    return result, state


def issue_refund(state: dict, amount: float, reason: str) -> tuple:
    """
    Issue a refund to the customer.
    """
    if amount <= 0:
        return {"error": "Refund amount must be greater than 0."}, state
    if amount > 500:
        return {"error": "Refund amounts above $500 require manager approval. Use escalate_to_human."}, state

    state["ticket"]["refund_issued"] = amount
    state["ticket"]["refund_reason"] = reason
    state["ticket"]["notes"].append(f"Refund issued: ${amount:.2f} — {reason}")

    result = {
        "success": True,
        "refund_amount": amount,
        "reason": reason,
        "processing_days": 3,
        "message": f"Refund of ${amount:.2f} initiated. Will appear in 3-5 business days.",
    }
    return result, state


def escalate_to_human(state: dict, reason: str) -> tuple:
    """
    Escalate the ticket to a human agent.
    """
    state["ticket"]["escalated"] = True
    state["ticket"]["escalation_reason"] = reason
    state["ticket"]["status"] = "escalated"
    state["ticket"]["notes"].append(f"Escalated to human: {reason}")

    result = {
        "success": True,
        "escalated": True,
        "reason": reason,
        "message": "Ticket escalated to senior support team. Customer will be contacted within 2 hours.",
    }
    return result, state


def close_ticket(state: dict, final_message: str) -> tuple:
    """
    Close the ticket with a final message to the customer.
    """
    state["ticket"]["status"] = "closed"
    state["ticket"]["resolved"] = True
    state["conversation"].append({
        "role": "agent",
        "content": final_message,
        "tone": "professional",
    })

    result = {
        "success": True,
        "ticket_closed": True,
        "final_message": final_message,
    }
    return result, state


# ---------------------------------------------------------------------------
# Tool router
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "search_kb":         lambda state, p: search_kb(state, p.get("query", "")),
    "get_order_details": lambda state, p: get_order_details(state, p.get("order_id", "")),
    "update_ticket":     lambda state, p: update_ticket(state, p.get("status", "in_progress"), p.get("note", "")),
    "send_reply":        lambda state, p: send_reply(state, p.get("message", ""), p.get("tone", "professional")),
    "ask_clarification": lambda state, p: ask_clarification(state, p.get("question", "")),
    "issue_refund":      lambda state, p: issue_refund(state, float(p.get("amount", 0)), p.get("reason", "")),
    "escalate_to_human": lambda state, p: escalate_to_human(state, p.get("reason", "")),
    "close_ticket":      lambda state, p: close_ticket(state, p.get("final_message", "Thank you for contacting us.")),
}

# All available tool names for reference
ALL_TOOLS = list(TOOL_MAP.keys())
