import base64
import json
from openai import OpenAI

# Videoyu yerel diskten okuyup metin formatına (base64) çeviriyoruz
def encode_video_to_base64(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode('utf-8')

video_yolu ="Videos/test2.mp4"
base64_video = encode_video_to_base64(video_yolu)

# vLLM Sunucuna Bağlan
# Göz Modeli İstemcisi (Port 8000)
client = OpenAI(
    api_key="TEKNOFEST-ALFAI",
    base_url="http://localhost:8000/v1",
)

# Beyin Modeli İstemcisi (Port 8001)
client_beyin = OpenAI(
    api_key="TEKNOFEST-ALFAI",
    base_url="http://localhost:8001/v1"
)

BEYIN_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"

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
    max_tokens=3000,
    temperature=0.1,
    response_format={"type": "json_object"}          
)

vision_output = response.choices[0].message.content

world_state = json.loads(vision_output)

print("\n===== WORLD STATE =====")
print(json.dumps(world_state, indent=2, ensure_ascii=False))


state = {
    "video_path": video_yolu,
    "vision": world_state,
    "context": {},
    "memory": {},
    "risk_analysis": {},
    "decision": None,
    "tool_outputs": {},
    "final_response": None
}
print("\n===== STATE =====")
print(json.dumps(state, indent=2, ensure_ascii=False))
print("🧠 AŞAMA 2: Beyin modeli yapılandırılmış veriyi devralıyor, karar mekanizması çalışıyor...\n")

# Ajanın alet çantası (Tools)
tools = [
    {
        "type": "function",
        "function": {
            "name": "saglik_ekibini_cagir",
            "description": "Sahada acil tıbbi müdahale gerektiren bir durum tespit edildiğinde sağlık ekibini çağırmak için kullanılır.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sebep": {
                        "type": "string",
                        "description": "Sağlık ekibinin çağrılma nedeni"
                    },
                },
                "required": ["sebep"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "guvenligi_uyar",
            "description": "Sahada güvenlik personelinin müdahale etmesi gereken durumlarda kullanılır.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mesaj": {
                        "type": "string",
                        "description": "Güvenlik personeline iletilecek uyarı mesajı"
                    }
                },
                "required": ["mesaj"]
            }
        }
    }
]

# Gözün ürettiği JSON metnini doğrudan Beynin promptuna enjekte ediyoruz
beyin_response = client_beyin.chat.completions.create(
    model=BEYIN_MODEL,
    messages=[
        {
    "role": "system",
    "content": """
Sen gerçek zamanlı çalışan otonom bir karar destek ajanısın.

Sana başka bir yapay zeka modeli tarafından oluşturulmuş yapılandırılmış saha analiz verisi verilecektir.

Görevin;

- olay akışını analiz etmek,
- risk seviyesini değerlendirmek,
- gerçekten müdahale gerekip gerekmediğine karar vermek,
- gerekiyorsa uygun aracı (tool) kullanmak,
- gerekmiyorsa kısa bir durum değerlendirmesi yapmaktır.

Kurallar:

- Sadece verilen analiz verisini kullan.
- Yeni olay üretme.
- Tahminde bulunma.
- Video hakkında varsayım yapma.
- Gereksiz tool çağırma.
- Aynı olay için yalnızca gerekli olan tool'u kullan.
- Sağlık açısından acil müdahale gerekiyorsa "saglik_ekibini_cagir" fonksiyonunu kullan.
- Güvenlik personelinin müdahalesi gerekiyorsa "guvenligi_uyar" fonksiyonunu kullan.
- Hiçbir aksiyon gerekmiyorsa tool kullanmadan kısa bir değerlendirme yaz.
"""
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
    tools=tools,
    tool_choice="auto",
    temperature=0.1
)


mesaj = beyin_response.choices[0].message

print("--- 🎯 NİHAİ SİSTEM ÇIKTISI ---")
if mesaj.tool_calls:
    print("🚨 SİSTEM BİR AKSİYON TETİKLEDİ (FONKSİYON KULLANILDI) 🚨")
    for tool in mesaj.tool_calls:
        print(f"-> Çalıştırılan Fonksiyon: {tool.function.name}")
        print(f"-> Üretilen Parametreler: {tool.function.arguments}")
else:
    print("📝 SADECE YANIT ÜRETİLDİ (Ajan tool kullanmaya gerek görmedi):")
    print(mesaj.content)

