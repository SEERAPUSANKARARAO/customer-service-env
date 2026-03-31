"""
env.py
Core CustomerServiceEnv — implements the OpenEnv step/reset/state interface.
"""

from app.tasks import TASK_REGISTRY
from app.graders import GRADER_REGISTRY
from app.tools import TOOL_MAP, ALL_TOOLS


class CustomerServiceEnv:
    """
    OpenEnv-compliant environment for customer service agent training.
    Supports three tasks: easy, medium, hard.
    All state is held in self._state (a plain dict) for easy serialization.
    """

    VALID_TASKS = {"easy", "medium", "hard"}

    def __init__(self):
        self._state: dict = {}
        self._grader = None
        self._available_tools: list = []

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def reset(self, task_id: str, seed: int = None) -> dict:
        """
        Initialize a new episode for the given task.
        Returns the first observation.
        """
        if task_id not in self.VALID_TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from: {self.VALID_TASKS}")

        make_state, tools = TASK_REGISTRY[task_id]
        self._grader = GRADER_REGISTRY[task_id]
        self._available_tools = list(tools)

        self._state = make_state(seed=seed)
        self._state["tools_used"] = []
        self._state["done"] = False

        return self._build_observation(reward=0.0, done=False, tool_result=None, info={})

    def step(self, action: dict) -> dict:
        """
        Execute one action in the environment.
        action: {"tool": str, "params": dict}
        Returns observation dict.
        """
        if not self._state:
            return self._error_obs("Environment not initialized. Call /reset first.")

        if self._state.get("done"):
            return self._error_obs("Episode is already done. Call /reset to start a new one.")

        tool = action.get("tool", "").strip()
        params = action.get("params", {})

        # Validate tool
        if tool not in self._available_tools:
            obs = self._build_observation(
                reward=-0.05,
                done=False,
                tool_result=None,
                info={"error": f"Tool '{tool}' is not available. Available: {self._available_tools}"},
            )
            self._state["step_count"] += 1
            return obs

        if tool not in TOOL_MAP:
            obs = self._build_observation(
                reward=-0.05,
                done=False,
                tool_result=None,
                info={"error": f"Unknown tool '{tool}'."},
            )
            self._state["step_count"] += 1
            return obs

        # Execute tool
        try:
            tool_result, self._state = TOOL_MAP[tool](self._state, params)
        except Exception as e:
            obs = self._build_observation(
                reward=-0.05,
                done=False,
                tool_result=None,
                info={"error": f"Tool execution error: {str(e)}"},
            )
            self._state["step_count"] += 1
            return obs

        self._state["tools_used"].append(tool)
        self._state["step_count"] += 1

        # Check terminal conditions
        done = self._is_done()
        self._state["done"] = done

        # Compute reward
        if done:
            grader_result = self._grader(self._state)
            reward = grader_result["score"]
            info = {"grader_breakdown": grader_result["breakdown"], "final_score": reward}
        else:
            reward = self._partial_reward()
            info = {"partial_reward": reward, "step": self._state["step_count"]}

        return self._build_observation(reward=reward, done=done, tool_result=tool_result, info=info)

    def state(self) -> dict:
        """Return the full internal state (for /state endpoint)."""
        return self._state

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _is_done(self) -> bool:
        ticket = self._state.get("ticket", {})
        step_count = self._state.get("step_count", 0)
        max_steps = self._state.get("max_steps", 10)

        return (
            ticket.get("resolved", False)
            or ticket.get("status") == "closed"
            or ticket.get("escalated", False)
            or step_count >= max_steps
        )

    def _partial_reward(self) -> float:
        """
        Give incremental reward signals during the episode (before done).
        This is the 'partial progress signal' judges look for.
        Rewards correct tool use as it happens.
        """
        tools_used = self._state.get("tools_used", [])
        task_id = self._state.get("task_id", "easy")
        score = 0.0

        if task_id == "easy":
            if "search_kb" in tools_used:                      score += 0.05
            if "send_reply" in tools_used:                     score += 0.05
            if "ask_clarification" in tools_used:              score += 0.02

        elif task_id == "medium":
            if "ask_clarification" in tools_used:              score += 0.05
            if "get_order_details" in tools_used:              score += 0.08
            if "send_reply" in tools_used:                     score += 0.05
            if "update_ticket" in tools_used:                  score += 0.03

        elif task_id == "hard":
            if "get_order_details" in tools_used:              score += 0.05
            if "send_reply" in tools_used:                     score += 0.05
            if "issue_refund" in tools_used:                   score += 0.08
            if "ask_clarification" in tools_used:              score += 0.02

        return round(min(score, 0.30), 4)  # partial reward capped at 0.30

    def _build_observation(self, reward: float, done: bool,
                           tool_result, info: dict) -> dict:
        return {
            "ticket": self._state.get("ticket", {}),
            "conversation": self._state.get("conversation", []),
            "available_tools": self._available_tools,
            "step_count": self._state.get("step_count", 0),
            "done": done,
            "reward": round(reward, 4),
            "tool_result": tool_result,
            "info": info,
        }

    def _error_obs(self, message: str) -> dict:
        return {
            "ticket": self._state.get("ticket", {}),
            "conversation": self._state.get("conversation", []),
            "available_tools": self._available_tools,
            "step_count": self._state.get("step_count", 0),
            "done": self._state.get("done", False),
            "reward": 0.0,
            "tool_result": None,
            "info": {"error": message},
        }
