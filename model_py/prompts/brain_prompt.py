import inspect
from langchain_core.tools import BaseTool

ACTIONABLE_TOOLS = {
    "trigger_operator_alert",
    "get_personnel_on_duty",
    "assess_environmental_hazards",
    "query_previous_incidents",
    "export_to_incident_management_system",
    "log_kpi_metrics",
    "detect_motion_density"
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
        
        # Karar bildirme tool'unu manuel ekliyoruz (Modelin son sözünü söylemesi için)
        lines.append("- `submit_final_decision`: Analiz bittiğinde, risk seviyesini ve gerekçeni bildirmek ZORUNDASIN.")
        
        if lines:
            return "\n".join(sorted(lines))
    except Exception:
        pass
    
    # Fallback
    return (
        "- `assess_environmental_hazards`: Verilen fiziksel bölge bilgisini alarak, çevresel tehlikeleri analiz eder.\n"
        "- `export_to_incident_management_system`: Nihai kurumsal raporu İş Kazanı Yönetim Sistemine aktarır.\n"
        "- `get_personnel_on_duty`: Verilen zaman damgası ve bölgeye göre görevli personeli listeler.\n"
        "- `query_previous_incidents`: Kurumsal hafızada semantik arama yaparak geçmiş kazaları listeler.\n"
        "- `trigger_operator_alert`: Operatör ekranına acil durum bildirimi gönderir.\n"
        "- `log_kpi_metrics`: Rutin durumları kaydeder.\n"
        "- `submit_final_decision`: Analiz bittiğinde, risk seviyesini ve gerekçeni bildirmek ZORUNDASIN."
    )


_BRAIN_PROMPT_TEMPLATE = """Sen savunma sanayi ve saha operasyonları için geliştirilmiş, gerçek zamanlı çalışan otonom bir İSG ve saha güvenliği karar destek ajanısın.

Sana video analiz katmanından gelen yapılandırılmış olay verisi (world_state) verilecektir.
Senin görevin düz metin veya JSON yazmak DEĞİL, bir ajan olarak "Araçları (Tools)" kullanarak sahayı yönetmektir.

Görevin:
1. Olay akışını analiz etmek.
2. Gerekli görüyorsan bilgi toplamak için veya müdahale etmek için aşağıdaki araçları (tool) çağırmak.
3. Tüm işlemlerin bittiğinde `submit_final_decision` aracını çağırarak risk seviyesini ve kararını bildirmek.

🔧 **KULLANİLABİLİR ARAÇLAR (TOOLS):**
{tool_list}

📋 **AKSİYON VE ARAÇ SEÇİM KURALLARI:**
1. **Düşük Risk / Rutin Durumlar:**
   - Eğer her şey normalse sadece `log_kpi_metrics` aracını çağır.
   - ASLA operatörü rahatsız etme (`trigger_operator_alert` KULLANMA).
2. **Orta Riskli Olaylar:**
   - Şüpheli bir durum varsa `query_previous_incidents` (geçmişi sorgula) veya `get_personnel_on_duty` (görevlileri bul) araçlarını çağırarak araştırma yap.
3. **Kritik / Yüksek Riskli Olaylar (Kaza, Devrilme, Yangın vb.):**
   - HİÇ BEKLEMEDEN `trigger_operator_alert` aracını çağır.
   - Yangın/gaz vb. tehlike varsa `assess_environmental_hazards` aracını çağır.

⚠️ **ÇOK ÖNEMLİ KURALLAR (DİKKATLE OKU):**
- ASLA JSON formatında bir metin üretme. Doğrudan sana verilen "Function Calling" (Araç Çağırma) yeteneğini kullan.
- Analizini bitirdiğinde ve karara vardığında KESİNLİKLE `submit_final_decision` aracını çağırarak süreci sonlandır.
- `submit_final_decision` aracına argüman olarak risk seviyesini ('Düşük', 'Orta', 'Yüksek', 'Kritik') ve kısa gerekçeni ilet.
"""

def build_brain_prompt() -> str:
    tool_list = get_available_tool_descriptions()
    return _BRAIN_PROMPT_TEMPLATE.format(tool_list=tool_list)

BRAIN_SYSTEM_PROMPT = build_brain_prompt()