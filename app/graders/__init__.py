from app.graders.grader_easy import grade as grade_easy
from app.graders.grader_medium import grade as grade_medium
from app.graders.grader_hard import grade as grade_hard

GRADER_REGISTRY = {
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
}
