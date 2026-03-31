from pydantic import BaseModel, Field
from typing import Any, Optional, List, Dict


class Action(BaseModel):
    tool: str = Field(..., description="Name of the tool to call")
    params: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")


class ResetRequest(BaseModel):
    task_id: str = Field(..., description="Task difficulty: 'easy' | 'medium' | 'hard'")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class CustomerInfo(BaseModel):
    name: str
    email: str
    tier: str = "regular"


class Ticket(BaseModel):
    id: str
    subject: str
    description: str
    status: str = "open"
    customer_info: Dict[str, Any]
    notes: List[str] = Field(default_factory=list)
    resolved: bool = False
    escalated: bool = False
    refund_issued: float = 0.0
    refund_reason: Optional[str] = None
    escalation_reason: Optional[str] = None


class Message(BaseModel):
    role: str   # "customer" | "agent"
    content: str
    tone: Optional[str] = None


class Observation(BaseModel):
    ticket: Dict[str, Any]
    conversation: List[Dict[str, Any]]
    available_tools: List[str]
    step_count: int
    done: bool
    reward: float
    tool_result: Optional[Any] = None
    info: Dict[str, Any] = Field(default_factory=dict)


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
