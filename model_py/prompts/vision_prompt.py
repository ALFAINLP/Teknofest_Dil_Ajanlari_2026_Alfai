JSON_SCHEMA = """
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

VISION_SYSTEM_PROMPT = f"""
Sen bir video analiz uzmanısın.

Videoyu analiz et ve KESİNLİKLE SADECE aşağıdaki JSON formatında,
Türkçe dilinde çıktı ver.

Başka hiçbir açıklama metni ekleme.

{JSON_SCHEMA}
"""

VISION_USER_PROMPT = """
Videoyu baştan sona analiz et.

Özellikle yengeci tutan adamın hareketlerine odaklan ve saniye saniye incele.

Gereksiz görsel detaylara
(renkler, arka plan bulanıklığı, çevresel nesneler)
girme.

Sadece temel eylemleri ve nesne hareketlerini
zaman damgalarıyla birlikte
kısa,
net
ve yapılandırılmış bir şekilde
Türkçe olarak özetle.

Risk durumunu belirle.

Çıktını verilen JSON şemasına tamamen uygun oluştur.

Başka hiçbir şey yazma.
"""
