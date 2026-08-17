import json
from model_py.services.client import brain_client
from model_py.config.settings import (
    BRAIN_MODEL,
    TEMPERATURE,
)
from model_py.tools.tool_definitions import TOOLS

def analyze_state(messages: list):
    """
    ReAct döngüsü boyunca gelen LangGraph mesaj geçmişini alır,
    OpenAI uyumlu sözlük yapısına dönüştürür ve Brain modeline iletir.
    """
    formatted_messages = []
    
    
    for msg in messages:
        # Eğer mesaj zaten bir sözlük ise doğrudan ekle
        if isinstance(msg, dict):
            formatted_messages.append(msg)
            continue
            
        # Eğer mesaj düz bir string ise sözlüğe çevirip ekle
        if isinstance(msg, str):
            formatted_messages.append({
                "role": "user",
                "content": msg
            })
            continue
            
        role = "user"
        if msg.__class__.__name__ == "AIMessage":
            role = "assistant"
        elif msg.__class__.__name__ == "SystemMessage":
            role = "system"
        elif msg.__class__.__name__ == "ToolMessage":
            role = "tool"
            
        msg_dict = {
            "role": role,
            "content": msg.content
        }
        
        # Eğer model bir araç çağırdıysa, araç çağrı detaylarını da OpenAI formatına eşle
        if role == "assistant" and getattr(msg, "tool_calls", None):
            msg_dict["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["args"])
                    }
                } for tc in msg.tool_calls
            ]
            
        if role == "tool" and hasattr(msg, "tool_call_id"):
            msg_dict["tool_call_id"] = msg.tool_call_id
            msg_dict["name"] = getattr(msg, "name", "tool")
            
        formatted_messages.append(msg_dict)

    response = brain_client.chat.completions.create(
        model=BRAIN_MODEL,
        messages=formatted_messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=TEMPERATURE,
    )

    return response.choices[0].message
