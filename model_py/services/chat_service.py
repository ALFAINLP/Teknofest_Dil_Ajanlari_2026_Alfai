"""
ALFAI sohbet servisi.
Llama 3.1'i araç çağrısı (tool_call) olmadan saf konuşma modunda kullanır.
Bağlam (video analizi + hafıza) sistem promptuna enjekte edilir;
model yalnızca metin yanıtı üretir.
"""
from model_py.services.client import brain_client
from model_py.config.settings import BRAIN_MODEL, TEMPERATURE
from model_py.prompts.chat_prompt import CHAT_SYSTEM_PROMPT
def chat(messages: list) -> str:
    """
    Verilen mesaj geçmişini Llama'ya gönderir ve yanıtı string olarak döndürür.
    Args:
        messages: [{"role": "system"|"user"|"assistant", "content": "..."}] listesi.
                  İlk eleman zaten sistem mesajı içermelidir (build_chat_messages tarafından eklenir).
    Returns:
        Llama'nın ürettiği yanıt metni. Boşsa fallback mesaj döner.
    """
    response = brain_client.chat.completions.create(
        model=BRAIN_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        # tools parametresi kasıtlı olarak verilmedi — saf konuşma modu
    )
    content = response.choices[0].message.content
    return content if content else "Yanıt üretilemedi."
def build_chat_messages(
    user_message: str,
    analysis_context: dict | None,
    memory: dict | None,
    history: list[dict] | None = None,
) -> list[dict]:
    """
    Sistem promptunu, bağlamı, geçmiş mesajları ve yeni kullanıcı mesajını
    birleştirerek Llama'ya gönderilebilecek mesaj listesi oluşturur.
    Args:
        user_message:      Operatörün sorusu.
        analysis_context:  Frontend'den gelen aktif analiz verisi (events, risk, vb.).
        memory:            memory.json'dan yüklenen geçmiş hafıza.
        history:           Önceki sohbet turu mesajları (opsiyonel, çoklu tur için).
    Returns:
        OpenAI mesaj formatında liste.
    """
    context_block = _build_context_block(analysis_context, memory)
    system_content = CHAT_SYSTEM_PROMPT + context_block
    messages: list[dict] = [{"role": "system", "content": system_content}]
    # Önceki sohbet turlarını ekle (çoklu tur desteği)
    if history:
        for turn in history:
            role = turn.get("role")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    # Yeni kullanıcı mesajını ekle
    messages.append({"role": "user", "content": user_message})
    return messages
def _build_context_block(analysis_context: dict | None, memory: dict | None) -> str:
    """
    Aktif analiz verisi ve hafızayı okunabilir bir metin bloğuna çevirir.
    Bu blok sistem promptunun sonuna eklenir.
    """
    if not analysis_context and not memory:
        return "\n\n[Bağlam: Henüz analiz yapılmamış.]"
    lines: list[str] = ["\n\n## Aktif Video Analizi Bağlamı"]
    if analysis_context:
        lines.append(f"- **Özet**: {analysis_context.get('summary', 'Yok')}")
        lines.append(f"- **Risk Seviyesi**: {analysis_context.get('risk', 'Bilinmiyor')}")
        lines.append(f"- **Risk Skoru**: %{analysis_context.get('risk_score', '?')}")
        lines.append(f"- **Risk Trendi**: {analysis_context.get('risk_trend', 'Bilinmiyor')}")
        actions = analysis_context.get("actions", [])
        if actions:
            lines.append(f"- **Önerilen Aksiyonlar**: {'; '.join(str(a) for a in actions)}")
        events = analysis_context.get("events", [])
        if events:
            lines.append("\n### Tespit Edilen Olaylar")
            for evt in events[:10]:  # En fazla 10 olay — prompt taşmasını önler
                time_label = evt.get("time", "?")
                event_text = evt.get("event", "Açıklama yok")
                risk_label = evt.get("risk", "")
                lines.append(f"  - [{time_label}] {event_text}" + (f" (Risk: {risk_label})" if risk_label else ""))
    if memory:
        event_history = memory.get("event_history", [])
        risk_history = memory.get("risk_history", [])
        lines.append("\n## Geçmiş Hafıza")
        lines.append(f"- Toplam geçmiş olay sayısı: {len(event_history)}")
        if risk_history:
            lines.append(f"- Geçmiş risk seviyeleri (son 5): {risk_history[-5:]}")
    return "\n".join(lines)
