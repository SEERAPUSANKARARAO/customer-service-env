from app.tasks.task_easy import make_state as make_easy_state, AVAILABLE_TOOLS as EASY_TOOLS
from app.tasks.task_medium import make_state as make_medium_state, AVAILABLE_TOOLS as MEDIUM_TOOLS
from app.tasks.task_hard import make_state as make_hard_state, AVAILABLE_TOOLS as HARD_TOOLS

TASK_REGISTRY = {
    "easy":   (make_easy_state,   EASY_TOOLS),
    "medium": (make_medium_state, MEDIUM_TOOLS),
    "hard":   (make_hard_state,   HARD_TOOLS),
}
