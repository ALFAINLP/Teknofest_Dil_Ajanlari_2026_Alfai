import json
from model_py.services.client import brain_client
from model_py.config.settings import (
    BRAIN_MODEL,
    TEMPERATURE,
)
from model_py.tools.tool_definitions import TOOLS
from model_py.prompts.brain_prompt import build_brain_prompt

def analyze_state(messages: list):
    """
    ReAct döngüsü boyunca gelen LangGraph mesaj geçmişini alır,
    dinamik sistem promptunu başa ekler, OpenAI uyumlu sözlük yapısına 
    dönüştürür ve Brain modeline iletir.
    """
    formatted_messages = []
    
    # 1. Dinamik Sistem Promptunu mesajların en başına ekle
    system_prompt_content = build_brain_prompt()
    formatted_messages.append({
        "role": "system",
        "content": system_prompt_content
    })
    
    for msg in messages:
        # Mesaj zaten system mesajıysa (eski prompt kalıntısı) dinamik prompt ile ezildiği için atla
        if isinstance(msg, dict) and msg.get("role") == "system":
            continue
        if msg.__class__.__name__ == "SystemMessage":
            continue

        # Eğer mesaj zaten bir sözlük ise ekle
        if isinstance(msg, dict):
            formatted_messages.append(msg)
            continue
            
        # Eğer mesaj düz bir string ise ekle
        if isinstance(msg, str):
            formatted_messages.append({
                "role": "user",
                "content": msg
            })
            continue
            
        role = "user"
        if msg.__class__.__name__ == "AIMessage":
            role = "assistant"
        elif msg.__class__.__name__ == "ToolMessage":
            role = "tool"
            
        msg_dict = {
            "role": role,
            "content": msg.content or ""
        }
        
        # Model bir araç çağırdıysa detayları ekle
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

    # 2. OpenAI API İsteği
    response = brain_client.chat.completions.create(
        model=BRAIN_MODEL,
        messages=formatted_messages,
        tools=TOOLS if TOOLS else None,
        tool_choice="auto" if TOOLS else None,
        temperature=TEMPERATURE,
        response_format={"type": "json_object"}  # JSON formatını kesinleştirir
    )

    return response.choices[0].message