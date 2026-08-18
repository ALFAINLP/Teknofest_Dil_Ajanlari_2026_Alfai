# model_py/tools/tool_definitions.py
import inspect
from typing import List, Dict, Any
from langchain_core.tools import BaseTool
import tools.mock_actions as mock_actions

def get_all_langchain_tools() -> List[BaseTool]:
    """mock_actions.py içerisindeki tüm @tool nesnelerini dinamik olarak toplar."""
    discovered_tools = []
    for name, obj in inspect.getmembers(mock_actions):
        if isinstance(obj, BaseTool):
            discovered_tools.append(obj)
    return discovered_tools

def get_openai_tools_schema() -> List[Dict[str, Any]]:
    """LangChain araçlarını OpenAI / vLLM tools formatına dönüştürür."""
    tools_schema = []
    for tool_obj in get_all_langchain_tools():
        # Raporlama ve telemetri gibi iç araçları modelin aksiyon listesinden ayırmak istersen filtreleyebilirsin
        schema = {
            "type": "function",
            "function": {
                "name": tool_obj.name,
                "description": tool_obj.description.strip().split("\n")[0] if tool_obj.description else "",
                "parameters": tool_obj.args_schema.model_json_schema() if tool_obj.args_schema else {
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