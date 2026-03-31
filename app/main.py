"""
main.py
FastAPI server exposing the OpenEnv API: /reset, /step, /state, /health.
Runs on port 7860 for Hugging Face Spaces.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.env import CustomerServiceEnv
from app.models import Action, ResetRequest

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Customer Service OpenEnv",
    description=(
        "A real-world OpenEnv environment for training AI customer support agents. "
        "Supports three task difficulties: easy (FAQ), medium (missing order), "
        "hard (angry refund with escalation). "
        "Implements the OpenEnv API: /reset, /step, /state."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single environment instance (stateful per session)
env = CustomerServiceEnv()


# ---------------------------------------------------------------------------
# OpenEnv required endpoints
# ---------------------------------------------------------------------------

@app.post("/reset", summary="Start a new episode")
def reset(req: ResetRequest):
    """
    Initialize a new episode.
    - task_id: "easy" | "medium" | "hard"
    - seed: optional int for reproducibility
    Returns the first observation.
    """
    try:
        obs = env.reset(task_id=req.task_id, seed=req.seed)
        return JSONResponse(content=obs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step", summary="Take an action in the environment")
def step(action: Action):
    """
    Execute one tool call as an action.
    - tool: name of the tool (e.g. "search_kb", "send_reply")
    - params: dict of tool parameters

    Returns observation with reward (0.0–1.0), done flag, and grader breakdown on completion.
    """
    try:
        obs = env.step({"tool": action.tool, "params": action.params})
        return JSONResponse(content=obs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@app.get("/state", summary="Get the full environment state")
def get_state():
    """
    Returns the full internal state dict.
    Useful for debugging and external monitoring.
    """
    try:
        return JSONResponse(content=env.state())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"State retrieval failed: {str(e)}")


# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check")
def health():
    """Required by HF Spaces for the deployment ping."""
    return {"status": "ok", "env": "customer-service-env", "version": "1.0.0"}


@app.get("/", summary="Environment info")
def root():
    """Returns environment metadata and available tools."""
    return {
        "name": "Customer Service OpenEnv",
        "version": "1.0.0",
        "tasks": ["easy", "medium", "hard"],
        "api_endpoints": ["/reset", "/step", "/state", "/health"],
        "available_tools": [
            "search_kb",
            "get_order_details",
            "ask_clarification",
            "send_reply",
            "update_ticket",
            "issue_refund",
            "escalate_to_human",
            "close_ticket",
        ],
        "reward_range": [0.0, 1.0],
    }


@app.get("/tasks", summary="List all tasks with details")
def list_tasks():
    """Returns info about all available tasks."""
    return {
        "tasks": [
            {
                "id": "easy",
                "name": "FAQ / Password Reset",
                "description": "Resolve a simple support ticket using the knowledge base. Max 5 steps.",
                "available_tools": ["search_kb", "ask_clarification", "send_reply", "update_ticket", "close_ticket"],
                "max_steps": 5,
                "reward_threshold": 0.7,
            },
            {
                "id": "medium",
                "name": "Missing Order Investigation",
                "description": "Ask for order ID, check order status, communicate result. Max 8 steps.",
                "available_tools": ["search_kb", "ask_clarification", "get_order_details",
                                    "send_reply", "update_ticket", "issue_refund", "close_ticket"],
                "max_steps": 8,
                "reward_threshold": 0.7,
            },
            {
                "id": "hard",
                "name": "Angry Refund with Escalation Decision",
                "description": "De-escalate angry customer, verify order, issue refund, decide on escalation. Max 10 steps.",
                "available_tools": ["search_kb", "ask_clarification", "get_order_details", "send_reply",
                                    "update_ticket", "issue_refund", "escalate_to_human", "close_ticket"],
                "max_steps": 10,
                "reward_threshold": 0.7,
            },
        ]
    }


# ---------------------------------------------------------------------------
# Entry point (local dev)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, reload=True)
