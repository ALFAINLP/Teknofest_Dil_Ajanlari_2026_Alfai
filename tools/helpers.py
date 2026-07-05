# Zaman dönüştürücü ve vLLM API çağrısı işlemlerini tek merkezde toplayan yardımcı fonksiyonlar.

import json
import httpx
from config.settings import settings

def _format_timestamp(time_seconds: int) -> str:
    """
    Toplam saniye bilgisini alıp otomatik olarak AA:GG (Dakika:Saniye) formatına çevirir.
    Örnek: 494 saniye -> "08:14"
    """
    minutes = time_seconds // 60
    seconds = time_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


def _call_vllm_api(system_prompt: str, user_prompt: str, model_name: str, max_tokens: int, video_path: str = None) -> str:
    """
    Yerel vLLM sunucusuyla olan tüm HTTP POST ağ trafiğini tek noktada yönetir.
    Hata yakalamayı, HTTP durum kontrollerini ve markdown JSON temizliğini otonom yapar.
    """
    vllm_url = f"{settings.VLLM_SERVER_URL}/v1/chat/completions"
    
    # Çoklu modallı (Video + Metin) veya Salt Metin durumuna göre payload yapısı
    user_content = []
    if video_path:
        user_content.append({"type": "video", "video_url": f"file:///{video_path}"})
    user_content.append({"type": "text", "text": user_prompt})

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.1,  # Nesnel/kararlı analiz standardı
        "max_tokens": max_tokens
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        # vLLM çıkarımları için zaman aşımı payı 60 saniye tutuldu
        with httpx.Client(timeout=60.0) as client:
            response = client.post(vllm_url, json=payload, headers=headers)
            
        if response.status_code == 200:
            raw_content = response.json()["choices"][0]["message"]["content"].strip()
            
            # Model çıktısında oluşabilecek markdown (```json ... ```) kalıplarını temizler
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:-3].strip()
            elif raw_content.startswith("```"):
                raw_content = raw_content[3:-3].strip()
                
            return raw_content
        else:
            return json.dumps({"error": f"vLLM Sunucu Hatası: Status {response.status_code}"})
    except Exception as e:
        return json.dumps({"error": f"Bağlantı veya Ayrıştırma Hatası: {str(e)}"})
