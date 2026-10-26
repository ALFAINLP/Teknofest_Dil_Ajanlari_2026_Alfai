# Ekip Adı: ALFAI 
---
**Otonom Video Analiz ve Karar Destek Sistemi**

> ALFAI, endüstriyel saha güvenliği ve operasyon izleme için geliştirilmiş çok katmanlı bir video analiz ve karar destek sistemidir. Sistem, gelen video girdisini iki ayrı yapay zekâ modeline sırayla işleterek yapılandırılmış bir risk raporu ve operatöre yönelik aksiyon önerileri üretir.

Sistemin temel çalışma mantığı şöyledir: bir Göz (Vision) modeli videoyu kare kare analiz ederek olayları, zaman damgalarını ve risk seviyesini JSON formatında çıkarır; ardından bir Beyin (Brain) modeli bu yapılandırılmış veriyi alarak araç çağrıları (tool calls) aracılığıyla derinlemesine karar analizi yapar ve nihai raporu üretir. İki model arasındaki orkestrasyon LangGraph tarafından yönetilir.

---
## Kullanılan Agentic Framework

LangGraph tercih edilmiştir. Tek bir LLM çağrısına dayalı basit pipeline yaklaşımı yerine, her node'un ayrı sorumluluğa sahip olduğu ve durumun (state) node'lar arasında akışla taşındığı graf tabanlı bir mimari kurulmuştur. Bu yapı sayesinde Vision ve Brain katmanları birbirinden bağımsız geliştirilebilmekte, ReAct döngüsü kontrollü şekilde çalıştırılabilmekte ve pipeline hataları izole edilebilmektedir. 

---

## Mimari Genel Bakış


Proje mimarisinde video analizi için iki model farklı amaçlar için kullanılmıştır.

```
Vision model(Qwen3-VL-32B-Instruct), video girdisi alarak önemli zaman adımlarını JSON formatında verir.

Brain model (Qwen/Qwen3.5-122B-A10B), video sonuçlarına göre LangGraph mimarisi ile uygun tool zincirini bulur.
```

*Her iki model de vLLM ile yerel ortamda servislenmiştir. Harici API bağımlılığı bulunmamaktadır.*



**Teknolojiler:**

| Katman | Teknoloji |
|---|---|
| Vision Modeli | Qwen3-VL-32B-Instruct (vLLM) |
| Brain Modeli | Qwen/Qwen3.5-122B-A10B (vLLM) |
| Ajan Orkestrasyonu | LangGraph |
| Model Servisleme | vLLM |
| API Katmanı | FastAPI |
| Şema Doğrulama | Pydantic v2 |
| Hafıza Yönetimi | JSON tabanlı MemoryManager |
| Benchmark | Özel KPI motoru (kpi_enhanced.py) |

---
## İmplemente Edilen Senaryolar ve Mock Fonksiyonlar

Sistem aşağıdaki 7 ana senaryo kategorisinde analiz yapabilmektedir:

- Endüstriyel İş Kazaları: Forklift devrilmesi, makineye sıkışma, yüksekten düşme, malzeme çökmesi
- Yangın ve Patlama: Fabrika yangını, akaryakıt istasyonu patlaması, elektrik ark patlaması, kundaklama
- Trafik Kazaları: Araç çarpışması, kavşak kazası, yaya ezilmesi, ambulans kazası
- Güvenlik İhlalleri: Silahlı saldırı, sokak darpı
- Kişisel Güvenlik: İzinsiz fiziksel temas, ofis içi taciz, fiziksel kavga
- Yapısal Hasar: Depo raf çökmesi, tavan yıkılması, enkaz oluşumu
- Rutin Operasyon: Normal trafik akışı, ofis çalışması, yaya geçişi

---
## Karşılaşılan Zorluklar ve Çözümler

- 
---
## Kurulum ve Çalıştırma

### Ön Koşullar

- **GPU:** NVIDIA A40 veya eşdeğeri (önerilen: 2× A40, toplam ~95 GB VRAM)
- **CUDA:** 12.x+
- **Python:** 3.10+
- **İşletim Sistemi:** Ubuntu 22.04 / 24.04

### Bağımlılıklar

```bash
# Sanal ortam oluştur
python3 -m venv .venv
source .venv/bin/activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

---

## Kullanım

### API Sunucusunu Başlat

```bash
uvicorn web_api:app --host 0.0.0.0 --port 8080 --reload
```

### NPM Başlat

```bash
npm run -- --host --127.0.0.1
```

### Örnek Çıktı

```json
{
  "summary": "Endüstriyel metal işleme tesisinde hidrolik sızıntı kaynaklı yangın çıkmış, çalışanlar tahliye edilmiştir.",
  "risk": "Kritik",
  "risk_score": 95,
  "events": [
    {
      "time": "00:05",
      "event": "Makineden çıkan alevler artmaya başladı.",
      "event_type": "KAZA",
      "phase": "baslangic",
      "risk": "Yüksek",
      "is_critical_moment": true,
      "confidence": "%95"
    },
    {
      "time": "00:22",
      "event": "Tesis içerisinde zincirleme patlama meydana geldi.",
      "event_type": "KAZA",
      "phase": "sonuc",
      "risk": "Kritik",
      "is_critical_moment": true,
      "confidence": "%99"
    }
  ],
  "actions": [
    "İtfaiye çağrılmalı ve tüm personel tahliye edilmeli",
    "Çevresel tehlike analizi yapılmalı, ikincil patlama riski kontrol edilmeli",
    "Olay olay yönetim sistemine kaydedilmeli"
  ],
  "kpi_metrics": {
    "ai_processing_status": "Kritik Müdahale Devrede",
    "confidence_score": 0.96,
    "detected_events_count": 5,
    "critical_events_count": 4,
    "critical_event_capture_rate": 0.8,
    "total_processing_seconds": 31.6
  }
}
```

### Benchmark Çalıştırma

```bash
python kpi_enhanced.py
# Çıktı: benchmark_results_enhanced.json
```

---

## Sistem Mimarisi

**Veri Akışı:**

1. **Vision Katmanı** — Video Base64 encode edilerek Qwen3-VL-32B'ye gönderilir. Model olayları, zaman damgalarını, risk seviyesini ve konumsal verileri (bbox/center) JSON formatında üretir. Hatalı JSON için 6 aşamalı onarım zinciri devreye girer.

2. **LangGraph Pipeline** — Üç node'lu lineer akış:
   - `extract_video_info_node`: Sahne metadata'sı, zone tespiti ve Vision çıktısının normalize edilmesi
   - `evaluate_risks_node`: Brain modeli ReAct döngüsünde araç zinciri kurar (max 5 adım). Kullanılabilir araçlar: `trigger_operator_alert`, `get_personnel_on_duty`, `assess_environmental_hazards`, `query_previous_incidents`, `detect_motion_density`, `log_kpi_metrics`, `export_to_incident_management_system`
   - `compile_report_node`: Pydantic şema doğrulaması, KPI hesaplama ve CMS export

3. **Hafıza Katmanı** — `MemoryManager` son 50 olayı ve 20 risk kaydını JSON'a persiste eder. Risk trendi (📈 Artan / 📉 Azalan / ⚠️ Kronik) geçmiş veriden hesaplanır.

4. **Benchmark Motoru** — `kpi_enhanced.py` ground truth karşılaştırması yapar: ±10 saniyelik toleranslı olay eşleştirmesi, Jaccard IoU tabanlı araç değerlendirmesi, katman bazlı analiz (tool seçimi, argüman çıkarımı, grounding).

**KPI Metrikleri:**

| Metrik | Açıklama |
|---|---|
| Event F1-Score | Olay tespit doğruluğu (precision × recall harmonik ortalaması) |
| Risk Doğruluğu | Doğru risk seviyesi tahmin oranı |
| Tool Jaccard (IoU) | Araç seti örtüşme oranı |
| Tool Exact Match | Araç setinin birebir eşleşme oranı |
| Real-Time Factor | İşlem süresi / video süresi (< 1.0 = gerçek zamandan hızlı) |

---

## VİDEO LİNKLERİ

•https://www.kaggle.com/datasets/minmints/ufc-crime-full-dataset/data


•https://www.youtube.com/watch?v=WjZB-g01JLE&list=PLz05EIQjBV2J8nRtbRgOouEJ0pAvdzPEq


•https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html

---

## İletişim ve Lisans

- **Takım Adı:** [ALFAI]
- **Lisans:** [Apache 2.0]

---


*TEKNOFEST 2026 Yapay Zeka Dil Ajanları Yarışması kapsamında geliştirilmiştir.*
