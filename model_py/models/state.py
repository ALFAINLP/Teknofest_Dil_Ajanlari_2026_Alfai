from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AgentState:
    video_path: str
    vision: Dict[str, Any]

    context: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    risk_analysis: Dict[str, Any] = field(default_factory=dict)
    decision: Optional[str] = None
    tool_outputs: Dict[str, Any] = field(default_factory=dict)
    final_response: Optional[str] = None
