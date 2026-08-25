# model_py/services/brain_service.py
import json
from model_py.services.client import brain_client
from model_py.config.settings import BRAIN_MODEL, TEMPERATURE
from model_py.tools.tool_definitions import TOOLS
from model_py.prompts.brain_prompt import BRAIN_SYSTEM_PROMPT

def sanitize_text(text):
    """Geçersiz Unicode (surrogate) karakterlerini temizleyerek sistemin çökmesini engeller."""
    if not isinstance(text, str):
        return text
    # 'ignore' parametresi bozuk karakterleri hata vermeden siler
    return text.encode('utf-8', 'ignore').decode('utf-8')

def analyze_state(messages: list, force_json: bool = False):
    """
    ReAct döngüsü mesaj geçmişini OpenAI / vLLM standardına çevirir.
    """
    formatted_messages = []
    
    # 1. En başa tek bir System Prompt ekle
    formatted_messages.append({
        "role": "system",
        "content": sanitize_text(BRAIN_SYSTEM_PROMPT)
    })
    
    for msg in messages:
        # State içinden gelen mükerrer system mesajlarını atla
        if isinstance(msg, dict):
            if msg.get("role") == "system":
                continue
            # 1. DEĞİŞİKLİK: Sözlük (dict) formatında gelen mesajları temizliyoruz
            safe_msg = dict(msg)
            if "content" in safe_msg:
                safe_msg["content"] = sanitize_text(safe_msg["content"])
            formatted_messages.append(safe_msg)
            continue
            
        if msg.__class__.__name__ == "SystemMessage":
            continue

        if isinstance(msg, str):
            # 2. DEĞİŞİKLİK: Düz string (str) formatında gelen mesajları temizliyoruz
            formatted_messages.append({"role": "user", "content": sanitize_text(msg)})
            continue
            
        role = "user"
        if msg.__class__.__name__ == "AIMessage":
            role = "assistant"
        elif msg.__class__.__name__ == "ToolMessage":
            role = "tool"
            
        # 3. DEĞİŞİKLİK: LangChain nesnelerinden (AIMessage vb.) gelen içerikleri temizliyoruz
        msg_dict = {"role": role, "content": sanitize_text(msg.content or "")}
        
        if role == "assistant" and getattr(msg, "tool_calls", None):
            msg_dict["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        # 4. DEĞİŞİKLİK: Tool argümanlarındaki (Vision'dan gelen metinler buraya düşer) bozuklukları temizliyoruz
                        "arguments": sanitize_text(json.dumps(tc["args"]) if isinstance(tc["args"], dict) else str(tc["args"]))
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
            frequency_penalty=0.5,
            temperature=0.0,  # Deterministik ve hatasız tool çağrısı için 0.0 önerilir
        )

    return response.choices[0].message