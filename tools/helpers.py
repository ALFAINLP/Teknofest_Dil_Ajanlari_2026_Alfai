# Zaman dönüştürücü ve vLLM API çağrısı işlemlerini tek merkezde toplayan yardımcı fonksiyonlar.

import json
import httpx
from config.settings import settings
import pathlib

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
    Model türüne göre (Qwen -> 8000, LLaMA -> 8001) doğru sunucu portuna yönlendirir.
    Hata yakalamayı, HTTP durum kontrollerini ve markdown JSON temizliğini otonom yapar.
    """
    if "qwen" in model_name.lower() or video_path is not None:
        # VISION portunu kesin olarak 8000'e sabitliyoruz
        base_url = getattr(settings, "VISION_BASE_URL", "http://localhost:8000/v1")
    else:
        base_url = getattr(settings, "BRAIN_BASE_URL", "http://localhost:8001/v1")

    if base_url.endswith("/v1"):
        vllm_url = f"{base_url}/chat/completions"
    else:
        vllm_url = f"{base_url}/v1/chat/completions"
    
   # Çoklu modallı (Video + Metin) veya Salt Metin durumuna göre payload yapısı
    # 🟢 DÜZELTME: vLLM'in beklediği katı role/content yapısına ve OpenAI standardına uyum
    user_content = []
    if video_path:
        # 🟢 DÜZELTME: Docker'ın beklediği sanal yola çevir
        # video_path tam bir yol olsa bile sadece dosya adını alıyoruz (örn: video.mp4)
        file_name = pathlib.Path(video_path).name 
        
        # Docker'a verdiğimiz izin klasörüne ekliyoruz (/tmp)
        docker_path = f"/tmp/{file_name}"
        
        user_content.append({
            "type": "video_url",
            # pathlib.Path.as_uri() otomatik olarak file:///tmp/... formatına çevirir
            "video_url": {"url": pathlib.Path(docker_path).as_uri()} 
        })
        
    user_content.append({"type": "text", "text": user_prompt})

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            # Eğer video yoksa direkt string gönderelim (bazı LLM'ler listeyi sevmez)
            {"role": "user", "content": user_content if video_path else user_prompt} 
        ],
        "temperature": 0.1,
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
