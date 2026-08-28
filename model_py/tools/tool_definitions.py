# model_py/tools/tool_definitions.py
from typing import List, Dict, Any
import tools.mock_actions as mock_actions

# Araç nesneleri. Sözlük sırası = şema sırası; anlamlı bir öncelik ifade ETMEZ,
# çağrım sırasına model karar verir (bkz. BRAIN_SYSTEM_PROMPT / GİRDİ-ÇIKTI).
_TOOL_OBJECTS = {
    "get_personnel_on_duty": mock_actions.get_personnel_on_duty,
    "assess_environmental_hazards": mock_actions.assess_environmental_hazards,
    "detect_motion_density": mock_actions.detect_motion_density,
    "query_previous_incidents": mock_actions.query_previous_incidents,
    "trigger_operator_alert": mock_actions.trigger_operator_alert,
    "export_to_incident_management_system": mock_actions.export_to_incident_management_system,
    "log_kpi_metrics": mock_actions.log_kpi_metrics,
    "submit_final_decision": mock_actions.submit_final_decision,
}

# Şema açıklamaları: TEK CÜMLE, aracın ne yaptığını söyler.
# Çağrılma koşulu, GİRDİ/ÇIKTI ve önkoşullar sistem promptunda tanımlıdır.
_TOOL_DESCRIPTIONS = {
    "get_personnel_on_duty":
        "Bölgedeki görevli personel sayısını, ilk yardım sertifikası durumunu "
        "ve en yakın müdahale süresini döner.",
    "assess_environmental_hazards":
        "Bölgedeki çevresel tehlike türünü, gaz/duman seviyesini ve yayılma "
        "riskini ölçer.",
    "detect_motion_density":
        "Belirtilen zaman aralığındaki ortalama hareket yoğunluğunu ölçer.",
    "query_previous_incidents":
        "Aynı bölgede geçmişte yaşanmış benzer olayları, kök nedenlerini ve "
        "alınan düzeltici aksiyonları döner.",
    "trigger_operator_alert":
        "Saha operatörüne acil müdahale bildirimi gönderir. Geri alınamaz.",
    "export_to_incident_management_system":
        "Olayı kurumsal olay yönetim sistemine kalıcı olarak kaydeder. "
        "Geri alınamaz.",
    "log_kpi_metrics":
        "Rutin izleme turunun performans kaydını yazar.",
    "submit_final_decision":
        "Nihai risk seviyesini ve gerekçesini bildirerek analizi sonlandırır.",
}

LANGCHAIN_TOOLS = list(_TOOL_OBJECTS.values())


def get_openai_tools_schema() -> List[Dict[str, Any]]:
    """
    OpenAI uyumlu tool şeması üretir.

    Açıklama kaynağı: _TOOL_DESCRIPTIONS (tek cümlelik tanım).
    Çağrılma koşulları ve önkoşullar BRAIN_SYSTEM_PROMPT içindedir;
    burada tekrar edilmez.
    """
    tools_schema = []
    for name, tool_obj in _TOOL_OBJECTS.items():
        description = _TOOL_DESCRIPTIONS.get(name)
        if not description:
            raise ValueError(f"_TOOL_DESCRIPTIONS içinde '{name}' tanımı yok.")

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