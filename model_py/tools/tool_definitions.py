# model_py/tools/tool_definitions.py
from typing import List, Dict, Any
import tools.mock_actions as mock_actions

# Araçları akıl yürütme önceliğine göre sıralıyoruz (İlk önce hareket ve olay sorgulama)
LANGCHAIN_TOOLS = [
    mock_actions.detect_motion_density,
    mock_actions.query_previous_incidents,
    mock_actions.get_personnel_on_duty,
    mock_actions.assess_environmental_hazards,
    mock_actions.trigger_operator_alert,
    mock_actions.log_kpi_metrics,
    mock_actions.export_to_incident_management_system,
    mock_actions.submit_final_decision
]

def get_openai_tools_schema() -> List[Dict[str, Any]]:
    """LangChain araçlarını eksiksiz docstring ile OpenAI / vLLM tools formatına dönüştürür."""
    tools_schema = []
    
    for tool_obj in LANGCHAIN_TOOLS:
        # Açıklamanın SADECE İLK SATIRINI DEĞİL, TAMAMINI ALIYORUZ
        doc_text = tool_obj.description.strip() if getattr(tool_obj, 'description', None) else ""
        
        schema = {
            "type": "function",
            "function": {
                "name": tool_obj.name,
                "description": doc_text,
                "parameters": tool_obj.args_schema.model_json_schema() if getattr(tool_obj, 'args_schema', None) else {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
        tools_schema.append(schema)
        
    return tools_schema

TOOLS = get_openai_tools_schema()