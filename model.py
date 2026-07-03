import base64
from openai import OpenAI

# Videoyu yerel diskten okuyup metin formatına (base64) çeviriyoruz
def encode_video_to_base64(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode('utf-8')

video_yolu ="Videos/test2.mp4"
base64_video = encode_video_to_base64(video_yolu)

# vLLM Sunucuna Bağlan
client = OpenAI(
    api_key="TEKNOFEST-ALFAI",
    base_url="http://localhost:8000/v1",
)

print(f"{video_yolu} okunuyor, yapılandırılmış JSON verisi üretiliyor...")

# Şartnamedeki "Mock Fonksiyon Örnekleri" yapısını doğrudan System prompt'a veriyoruz
json_semasi = """
{
  "summary": "Videoda gerçekleşen olayların Türkçe genel özeti",
  "events": [
    {"time": "00:00", "event": "Gerçekleşen 1. eylem"},
    {"time": "00:05", "event": "Gerçekleşen 2. eylem"}
  ],
  "risk": "Düşük/Orta/Yüksek (Duruma göre belirle)",
  "actions": [
    "Operatöre veya sahaya önerilen 1. aksiyon",
    "Operatöre veya sahaya önerilen 2. aksiyon"
  ]
}
"""

response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-8B-Instruct",
    messages=[
        {
            "role": "system",
            "content": f"Sen bir video analiz uzmanısın. Videoyu analiz et ve KESİNLİKLE SADECE aşağıdaki JSON formatında, Türkçe dilinde çıktı ver. Başka hiçbir açıklama metni ekleme:\n{json_semasi}"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "Videoyu baştan sona analiz et. Özellikle yengeci tutan adamın hareketlerine odaklan ve saniye saniye incele. Gereksiz görsel detaylara (renkler, arka plan bulanıklığı, çevresel nesneler) girme. Sadece temel eylemleri ve nesne hareketlerini zaman damgalarıyla birlikte kısa, net ve yapılandırılmış bir şekilde Türkçe olarak özetle ve risk durumunu belirle. Lütfen çıktını verilen JSON şemasına tamamen uygun olarak oluştur ve başka hiçbir şey yazma."
                },
                {
                    "type": "video_url",
                    "video_url": {
                        "url": f"data:video/mp4;base64,{base64_video}"
                    }
                }
            ],
        }
    ],
    max_tokens=1500,
    temperature=0.1,
    response_format={"type": "json_object"}          
)

print("\n--- JSON ÇIKTISI ---")
print(response.choices[0].message.content)
print(response.choices[0].message.content)
