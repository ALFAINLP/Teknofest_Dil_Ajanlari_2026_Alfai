import json
from model_py.services.client import brain_client
from model_py.config.settings import (
    BRAIN_MODEL,
    TEMPERATURE,
)
from model_py.prompts.brain_prompt import BRAIN_SYSTEM_PROMPT
from model_py.tools.tool_definitions import TOOLS

def analyze_state(world_state):
    """
    Vision modelinden gelen yapılandırılmış veriyi
    Brain modeline göndererek karar mekanizmasını çalıştırır.
    """

    response = brain_client.chat.completions.create(
        model=BRAIN_MODEL,
        messages=[
            {
                "role": "system",
                "content": BRAIN_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""
Aşağıda görüntü analiz modülünden elde edilen yapılandırılmış saha verisi bulunmaktadır.

Bu veriyi inceleyerek:

- olay akışını değerlendir,
- risk seviyesini göz önünde bulundur,
- gerçekten müdahale gerekip gerekmediğine karar ver,
- gerekiyorsa uygun tool'u kullan,
- gerekmiyorsa kısa bir değerlendirme yap.

Analiz verisi:

{json.dumps(world_state, ensure_ascii=False, indent=2)}
"""
            }
        ],
        tools=TOOLS,
        tool_choice="auto",
        temperature=TEMPERATURE,
    )

    return response.choices[0].message
