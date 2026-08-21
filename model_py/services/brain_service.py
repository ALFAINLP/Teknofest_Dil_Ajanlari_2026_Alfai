# model_py/services/brain_service.py
import json
from model_py.services.client import brain_client
from model_py.config.settings import BRAIN_MODEL, TEMPERATURE
from model_py.tools.tool_definitions import TOOLS
from model_py.prompts.brain_prompt import BRAIN_SYSTEM_PROMPT

def analyze_state(messages: list, force_json: bool = False):
    """
    ReAct döngüsü mesaj geçmişini OpenAI / vLLM standardına çevirir.
    """
    formatted_messages = []
    
    # 1. En başa tek bir System Prompt ekle
    formatted_messages.append({
        "role": "system",
        "content": BRAIN_SYSTEM_PROMPT
    })
    
    for msg in messages:
        # State içinden gelen mükerrer system mesajlarını atla
        if isinstance(msg, dict):
            if msg.get("role") == "system":
                continue
            formatted_messages.append(msg)
            continue
            
        if msg.__class__.__name__ == "SystemMessage":
            continue

        if isinstance(msg, str):
            formatted_messages.append({"role": "user", "content": msg})
            continue
            
        role = "user"
        if msg.__class__.__name__ == "AIMessage":
            role = "assistant"
        elif msg.__class__.__name__ == "ToolMessage":
            role = "tool"
            
        msg_dict = {"role": role, "content": msg.content or ""}
        
        if role == "assistant" and getattr(msg, "tool_calls", None):
            msg_dict["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["args"]) if isinstance(tc["args"], dict) else str(tc["args"])
                    }
                } for tc in msg.tool_calls
            ]
            
        if role == "tool" and hasattr(msg, "tool_call_id"):
            msg_dict["tool_call_id"] = msg.tool_call_id
            msg_dict["name"] = getattr(msg, "name", "tool")
            
        formatted_messages.append(msg_dict)

    # 2. OpenAI / vLLM API İsteği
    if force_json:
        response = brain_client.chat.completions.create(
            model=BRAIN_MODEL,
            messages=formatted_messages,
            tools=None,
            tool_choice=None,
            temperature=TEMPERATURE,
            response_format={"type": "json_object"}
        )
    else:
        active_tools = TOOLS if TOOLS else None
        response = brain_client.chat.completions.create(
            model=BRAIN_MODEL,
            messages=formatted_messages,
            tools=active_tools,
            tool_choice="auto" if active_tools else None,
            temperature=0.0,  # Deterministik ve hatasız tool çağrısı için 0.0 önerilir
        )

    return response.choices[0].message