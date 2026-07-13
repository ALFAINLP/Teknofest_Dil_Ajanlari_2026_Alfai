from typing import TypedDict, List, Dict, Any, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    video_path: str
    current_time_segments: List[int]
    detected_events: List[Dict[str, Any]]
    risk_assessments: List[str]
    actions_recommended: List[str]
    final_json_output: str
    next_step: str
    # ReAct için zorunlu Hafıza (Conversational Memory) alanı:
    messages: Annotated[list, add_messages]
