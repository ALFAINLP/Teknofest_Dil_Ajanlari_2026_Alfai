# main.py
import json
import os
from agent.graph import app

def main():
    # Adım 1: Sadece video girdi yolunu belirle (Şartname girdi gereksinimi)
    video_file_path = "Videos/test2.mp4"
    
    print("\n" + "="*60)
    print("TEKNOFEST ALFA-AI OTONOM ReAct KARAR DESTEK SİSTEMİ BAŞLATILIYOR")
    print("="*60)
    
    # Adım 2: LangGraph ReAct Ajanı için temiz başlangıç durumunu (State) hazırla
    # Artık telemetri, optimizasyon ve kritik saniye tarama adımlarını burada statik çağırmıyoruz!
    # Ajan (Brain) döngü içinde bu araçları kendisi otonom tetikleyecek.
    initial_state = {
        "video_path": video_file_path,
        "current_time_segments": [],
        "detected_events": [],
        "risk_assessments": [],
        "actions_recommended": [],
        "final_json_output": "",
        "next_step": "",
        "messages": [] # ReAct hafızasını boş ve temiz bir şekilde başlatıyoruz
    }
    
    # Adım 3: LangGraph Otonom ReAct Döngüsünü Tetikle
    print("\n" + "-"*60)
    print("[LangGraph]: Otonom ReAct Akıl Yürütme ve Karar Döngüsü Başlatıldı...")
    print(" -> Ajan sırasıyla donanımı ölçecek, parametre optimize edecek ve")
    print("    video analiz araçlarını otonom olarak sırasıyla tetikleyecektir.")
    print("-"*60)
    
    final_output = app.invoke(initial_state)
    
    # Adım 4: Döngü Sonu İletilerini ve Kararları Konsola Bas
    print("\n" + "="*60)
    print("ReAct DÖNGÜSÜ TAMAMLANDI: ADIM ADIM YOL HARİTASI (HAFIZA)")
    print("="*60 + "\n")
    
    # Modelin süreç boyunca kendi kendine aldığı kararları ve araç çıktılarını yazdırıyoruz
    for i, msg in enumerate(final_output.get("messages", [])):
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])
        
        print(f"[{i+1}] {role}:")
        if content:
            print(f"    Mesaj: {content}")
        if tool_calls:
            for tc in tool_calls:
                print(f"    🛠️ Eylem (Araç Çağrısı): {tc['name']} -> Argümanlar: {tc['args']}")
        print("-" * 40)
        
    # Adım 5: Şartnameye %100 Uyumlu, Pydantic Onaylı Çıktıyı Ekrana Bas
    print("\n" + "="*60)
    print("ŞARTNAMEYE UYGUN NİHAİ YAPILANDIRILMIŞ ÇIKTI (JSON)")
    print("="*60 + "\n")
    
    # Ajanın döngü sonunda otonom olarak ürettiği nihai raporu çekiyoruz
    final_json = final_output.get("final_json_output", "")
    
    if not final_json:
        # Eğer state'e doğrudan yazılmadıysa, mesaj geçmişindeki en son başarılı araç cevabını veya asistan mesajını süzebiliriz
        for msg in reversed(final_output.get("messages", [])):
            if msg.get("role") == "assistant" and msg.get("content"):
                final_json = msg["content"]
                break
                
    try:
        # JSON çıktısını güzelleştirerek yazdır
        parsed_report = json.loads(final_json)
        print(json.dumps(parsed_report, ensure_ascii=False, indent=2))
    except (json.JSONDecodeError, TypeError):
        # Fallback durumunda ham veriyi bas
        if final_json:
            print(final_json)
        else:
            print("HATA: ReAct döngüsü sonunda geçerli bir nihai rapor üretilemedi.")
            
    print("\n" + "="*60)
    print(" İşlem bitti. Tüm kararlar, optimizasyonlar ve raporlama otonom olarak tamamlandı.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
