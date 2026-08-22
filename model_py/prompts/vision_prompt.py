JSON_SCHEMA = """
{
  "summary": "Videoda gerçekleşen olayların Türkçe genel özeti",
  "events": [
    {
      "time": "MM:SS",
      "start_time": "MM:SS",
      "end_time": "MM:SS",
      "event": "Kısa eylem açıklaması",
      "event_type": "Olay türü",
      "phase": "başlangıç veya gelişim veya sonuç",
      "zone": "Olayın görüldüğü bölge",
      "center": {"x": 0.50, "y": 0.50},
      "bbox": {"x1": 0.35, "y1": 0.25, "x2": 0.65, "y2": 0.80},
      "risk": "Düşük/Orta/Yüksek/Kritik",
      "is_critical_moment": false,
      "confidence": 0.85
    }
  ],
  "risk": "Düşük/Orta/Yüksek/Kritik",
  "risk_score": 50,
  "actions": [
    "Risk seviyesine göre 1-5 adet operatör aksiyonu"
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
Videoyu baştan sona saniye saniye dikkatli biçimde analiz et.

Tüm zaman damgalarını MM:SS formatında yaz. Örnek: 00:03, 01:22, 02:45.
HH:MM:SS formatı kullanma.

İnsan, araç ve nesne hareketlerini zaman sırasına göre incele.
Her türlü hareketi ve eylemi events listesine MUTLAKA ekle:
- Kişi geçişi, yürüyüş, koşu, duruş, oturma
- Araç hareketi, durması, kapı açılması
- Nesneyle etkileşim, taşıma, bırakma
- Yön değişikliği, ani hareket, toplanma
- Çevresel değişim: ışık değişimi, kapı/geçit açılması

Videoda hareket eden veya değişen HER ŞEY bir olaydır. Boş events döndürme.

Her olay için start_time ve end_time belirt.
Her olay için phase alanını olay içeriğine göre belirle (sıraya göre değil): "başlangıç" ilk tetiklenme anı, "gelişim" yayılma/kötüleşme süreci, "sonuç" doruk veya bitiş anıdır. Aynı fazda birden fazla olay olabilir.
Kritik bir an varsa is_critical_moment true yap.

Her olayın temel öznesinin görüntü karesi üzerindeki konumunu belirt.
center alanındaki x ve y değerleri ile bbox alanındaki x1, y1, x2, y2
değerlerini 0.0 ile 1.0 arasında normalize edilmiş oranlar olarak yaz.
Sol üst köşe 0,0; sağ alt köşe 1,1 kabul edilir.
Konum güvenilir biçimde belirlenemiyorsa center ve bbox alanlarını null yaz;
kesinlikle rastgele koordinat uydurma.

Gereksiz görsel detaylara girme: renkler, arka plan, kamera açısı.
Sadece temel eylemleri ve hareketleri kısa ve net Türkçe ile yaz.
Videoda görülmeyen bilgi uydurma.

Tespit edilen olaylara bağlı en az iki uygulanabilir aksiyon öner. 
Gerektiğinde daha fazla aksiyon önerisinde bulunabilirsin.

Çıktını verilen JSON şemasına tamamen uygun oluştur.
Başka hiçbir şey yazma.
"""

