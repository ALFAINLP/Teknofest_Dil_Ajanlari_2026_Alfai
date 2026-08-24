from typing import TypedDict, List, Dict, Any, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Girdi
    video_path: str

    # Vision çıktısı
    current_time_segments: List[int]
    detected_events: List[Dict[str, Any]]
    risk_assessments: List[str]
    actions_recommended: List[str]

    # ── EKSİK OLAN ALANLAR (LangGraph node'lar arası taşınmıyor) ──
    vision_risk: str
    vision_risk_score: float
    risk_score: int
    video_summary: str
    technical_summary: str
    # ──────────────────────────────────────────────────────────────

    # Trend
    risk_trend: str

    # Memory
    event_history: List[Dict[str, Any]]
    tool_history: List[str]
    risk_history: List[str]

    # Temporal Context
    context_summary: Dict[str, Any]

    # Çıktı
    final_json_output: str
    next_step: str

    # LangGraph mesaj hafızası
    messages: Annotated[list, add_messages]

    current_zone: str