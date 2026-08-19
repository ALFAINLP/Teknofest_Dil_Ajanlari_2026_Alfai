from typing import List, Dict, Any
import tools.mock_actions as mock_actions

# 1. Araçları KESİN VE NET bir şekilde listeliyoruz. 
# inspect kullanmıyoruz, böylece BaseTool sınıf uyuşmazlığı veya Circular Import hatalarını kökten çözüyoruz.
LANGCHAIN_TOOLS = [
    mock_actions.assess_environmental_hazards,
    mock_actions.export_to_incident_management_system,
    mock_actions.get_personnel_on_duty,
    mock_actions.query_previous_incidents,
    mock_actions.trigger_operator_alert,
    mock_actions.log_kpi_metrics,
    mock_actions.detect_motion_density
]

def get_openai_tools_schema() -> List[Dict[str, Any]]:
    """LangChain araçlarını OpenAI / vLLM tools formatına dönüştürür."""
    tools_schema = []
    
    for tool_obj in LANGCHAIN_TOOLS:
        schema = {
            "type": "function",
            "function": {
                "name": tool_obj.name,
                # Description'ın sadece ilk satırını alır, hata vermemesi için boş string koruması var
                "description": tool_obj.description.strip().split("\n")[0] if getattr(tool_obj, 'description', None) else "",
                # Langchain tool argümanlarını JSON schema'ya çevirir
                "parameters": tool_obj.args_schema.model_json_schema() if getattr(tool_obj, 'args_schema', None) else {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
        tools_schema.append(schema)
        
    return tools_schema

# Servislerin ve LLM çağrılarının kullanacağı dinamik tool listesi
TOOLS = get_openai_tools_schema()