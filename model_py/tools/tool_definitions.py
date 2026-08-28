# model_py/tools/tool_definitions.py
from typing import List, Dict, Any
import tools.mock_actions as mock_actions
from model_py.prompts.brain_prompt import TOOL_CONSTRAINTS, TOOL_ORDER

# Sıra TOOL_ORDER'dan geliyor; burada elle sıralama YAPMA.
_TOOL_OBJECTS = {
    "trigger_operator_alert": mock_actions.trigger_operator_alert,
    "detect_motion_density": mock_actions.detect_motion_density,
    "assess_environmental_hazards": mock_actions.assess_environmental_hazards,
    "query_previous_incidents": mock_actions.query_previous_incidents,
    "get_personnel_on_duty": mock_actions.get_personnel_on_duty,
    "log_kpi_metrics": mock_actions.log_kpi_metrics,
    "export_to_incident_management_system": mock_actions.export_to_incident_management_system,
    "submit_final_decision": mock_actions.submit_final_decision,
}

# LangGraph'ın ToolNode'u için de aynı sıra
LANGCHAIN_TOOLS = [_TOOL_OBJECTS[name] for name in TOOL_ORDER]


def get_openai_tools_schema() -> List[Dict[str, Any]]:
    """
    Tool açıklaması olarak docstring DEĞİL, TOOL_CONSTRAINTS kullanılır.
    Tek gerçek kaynak: prompts/brain_prompt.py
    """
    tools_schema = []

    for name in TOOL_ORDER:
        tool_obj = _TOOL_OBJECTS[name]

        description = TOOL_CONSTRAINTS.get(name)
        if not description:
            raise ValueError(f"TOOL_CONSTRAINTS içinde '{name}' tanımı yok.")

        params = (
            tool_obj.args_schema.model_json_schema()
            if getattr(tool_obj, "args_schema", None)
            else {"type": "object", "properties": {}, "required": []}
        )

        tools_schema.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": params,
            }
        })

    return tools_schema


TOOLS = get_openai_tools_schema()