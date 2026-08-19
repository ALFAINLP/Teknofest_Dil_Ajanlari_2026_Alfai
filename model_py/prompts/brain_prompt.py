import inspect
from langchain_core.tools import BaseTool

ACTIONABLE_TOOLS = {
    "trigger_operator_alert",
    "get_personnel_on_duty",
    "assess_environmental_hazards",
    "query_previous_incidents",
    "export_to_incident_management_system"
}

def get_available_tool_descriptions() -> str:
    """ACTIONABLE_TOOLS setindeki araçların açıklamalarını döndürür.
    Circular import'u önlemek için mock_actions lazy import edilir."""
    try:
        import tools.mock_actions as mock_actions
        lines = []
        for name, obj in inspect.getmembers(mock_actions):
            if isinstance(obj, BaseTool) and obj.name in ACTIONABLE_TOOLS:
                doc = (obj.description or "Açıklama yok.").strip().split("\n")[0]
                lines.append(f"- `{obj.name}`: {doc}")
        if lines:
            return "\n".join(sorted(lines))
    except Exception:
        pass
    # Fallback: sabit liste (circular import veya import hatasında)
    return (
        "- `assess_environmental_hazards`: Verilen fiziksel bölge bilgisini alarak, çevresel tehlikeleri analiz eder.\n"
        "- `export_to_incident_management_system`: Nihai kurumsal raporu İş Kazanı Yönetim Sistemine aktarır.\n"
        "- `get_personnel_on_duty`: Verilen zaman damgası ve bölgeye göre görevli personeli listeler.\n"
        "- `query_previous_incidents`: Kurumsal hafızada semantik arama yaparak geçmiş kazaları listeler.\n"
        "- `trigger_operator_alert`: Operatör ekranına acil durum bildirimi gönderir."
    )

# Kuralları çok daha sert ve mantıksal (IF-THEN) olarak güncelledik.
_BRAIN_PROMPT_TEMPLATE = """Sen gerçek zamanlı çalışan otonom bir İSG ve saha güvenliği karar destek ajanısın.
Sana video analiz katmanından gelen yapılandırılmış olay verisi (world_state) verilecektir.

Görevin:
1. Olay akışını analiz etmek,
2. Risk seviyesini belirlemek ('Düşük', 'Orta', 'Yüksek', 'Kritik'),
3. Durumun gerektirdiği aksiyonları YALNIZCA aşağıdaki standart fonksiyon havuzundan seçmek,
4. Çıktıyı KESİNLİKLE belirtilen JSON formatında üretmek.

🔧 **KULLANİLABİLİR FONKSİYON HAVUZU:**
{tool_list}
- `log_kpi_metrics`: Rutin, normal ve düşük riskli durumları sisteme kaydeder.
- `detect_motion_density`: Hareket yoğunluğu, kaçışma veya kinetik anormallikleri analiz eder.

📋 **STANDART AKSİYON SEÇİM KURALLARI (ZORUNLU):**
1. **Düşük Risk / Rutin Durumlar (KRİTİK KURAL):**
   - EĞER tespit edilen risk "Düşük" ise, 'actions' listesi KESİNLİKLE VE SADECE ["log_kpi_metrics"] olmalıdır.
   - "Düşük" riskli durumlarda ASLA `export_to_incident_management_system` veya `trigger_operator_alert` KULLANILAMAZ.
2. **Orta Riskli Olaylar:**
   - Şüpheli davranış (araç kurCalama), yetkisiz alana yaklaşma gibi durumlarda risk en az "Orta" olmalı ve `query_previous_incidents` veya `get_personnel_on_duty` kullanılmalıdır.
3. **Kritik / Yüksek Riskli Olaylar (Kaza, Devrilme, Çarpışma, Yangın, Patlama vb.):**
   - Her kritik olayda mutlaka `trigger_operator_alert` ve `export_to_incident_management_system` aksiyonlarını ekle.
   - Yangın, gaz, patlama tehlikelerinde ek olarak: `assess_environmental_hazards` ekle.

📌 **ZORUNLU ÇIKTI FORMATı (JSON):**
{{
  "risk": "Düşük" | "Orta" | "Yüksek" | "Kritik",
  "actions": [
    "secilen_fonksiyon_1",
    "secilen_fonksiyon_2"
  ],
  "decision": "Kararın kısa Türkçe gerekçesi."
}}

⚠️ **YASAKLAR:**
- 'dispatch_health_team', 'notify_fire_department', 'normal_izleme' gibi listede olmayan fonksiyon isimleri ASLA uydurma.
- 'actions' içine açıklama metni yazma, sadece yukarıdaki havuzdaki fonksiyon adlarını kullan.
"""

def build_brain_prompt() -> str:
    """Her çağrıda dinamik tool listesini üretir ve prompt'a enjekte eder."""
    tool_list = get_available_tool_descriptions()
    return _BRAIN_PROMPT_TEMPLATE.format(tool_list=tool_list)

# Modül-level BRAIN_SYSTEM_PROMPT: evaluate_risks_node'da import ediliyorsa bu
# hâlâ circular import sorununa yol açabilir. build_brain_prompt() tercih edin.
BRAIN_SYSTEM_PROMPT = build_brain_prompt()