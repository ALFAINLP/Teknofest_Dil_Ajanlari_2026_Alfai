# main.py
import json
import os

from agent.graph import app
from agent.memory_manager import MemoryManager


def main():
    # Adım 1: Sadece video girdi yolunu belirle (Şartname girdi gereksinimi)
    video_file_path = "Videos/test2.mp4"

    print("\n" + "=" * 60)
    print("TEKNOFEST ALFA-AI OTONOM ReAct KARAR DESTEK SİSTEMİ BAŞLATILIYOR")
    print("=" * 60)

    # Daha önce kaydedilen hafızayı yükle
    memory = MemoryManager.load_memory()

    # Adım 2: LangGraph ReAct Ajanı için başlangıç durumunu hazırla
    initial_state = {
        "video_path": video_file_path,
        "current_time_segments": [],
        "detected_events": [],
        "risk_assessments": [],
        "actions_recommended": [],
        "final_json_output": "",
        "next_step": "",

        # Agent Memory (Kalıcı Hafıza)
        "event_history": memory["event_history"],
        "risk_history": memory["risk_history"],
        "tool_history": memory["tool_history"],

        # LangGraph Message Memory
        "messages": []
    }

    # Adım 3: LangGraph Otonom ReAct Döngüsünü Tetikle
    print("\n" + "-" * 60)
    print("[LangGraph]: Otonom ReAct Akıl Yürütme ve Karar Döngüsü Başlatıldı...")
    print(" -> Ajan sırasıyla donanımı ölçecek, parametre optimize edecek ve")
    print("    video analiz araçlarını otonom olarak sırasıyla tetikleyecektir.")
    print("-" * 60)

    final_output = app.invoke(initial_state)

    # Adım 4: Döngü Sonu İletilerini ve Kararları Konsola Bas
    print("\n" + "=" * 60)
    print("ReAct DÖNGÜSÜ TAMAMLANDI: ADIM ADIM YOL HARİTASI (HAFIZA)")
    print("=" * 60 + "\n")

    # Modelin süreç boyunca kendi kendine aldığı kararları ve araç çıktılarını yazdırıyoruz
    for i, msg in enumerate(final_output.get("messages", [])):
        
        # YENİ KISIM: dict (.get) yerine obje metotlarını kullanıyoruz
        role = getattr(msg, "type", "unknown").upper() # 'ai' döner, biz onu AI yaparız
        if role == "AI": role = "ASSISTANT" # Görüntü güzelliği için ufak bir dokunuş
            
        content = getattr(msg, "content", "")
        tool_calls = getattr(msg, "tool_calls", [])
        
        print(f"[{i+1}] {role}:")
        if content:
            print(f"    Mesaj: {content}")
            
        if tool_calls:
            for tc in tool_calls:
                print(f"    🛠️ Eylem (Araç Çağrısı): {tc['name']} -> Argümanlar: {tc['args']}")
        print("-" * 40)

    # Adım 5: Şartnameye %100 Uyumlu, Pydantic Onaylı Çıktıyı Ekrana Bas
    print("\n" + "=" * 60)
    print("ŞARTNAMEYE UYGUN NİHAİ YAPILANDIRILMIŞ ÇIKTI (JSON)")
    print("=" * 60 + "\n")

    final_json = final_output.get("final_json_output", "")

    if not final_json:
        for msg in reversed(final_output.get("messages", [])):
            if msg.get("role") == "assistant" and msg.get("content"):
                final_json = msg["content"]
                break

    try:
        parsed_report = json.loads(final_json)
        print(json.dumps(parsed_report, ensure_ascii=False, indent=2))
    except (json.JSONDecodeError, TypeError):
        if final_json:
            print(final_json)
        else:
            print("HATA: ReAct döngüsü sonunda geçerli bir nihai rapor üretilemedi.")

    print("\n" + "=" * 60)
    print("İşlem bitti. Tüm kararlar, optimizasyonlar ve raporlama otonom olarak tamamlandı.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
