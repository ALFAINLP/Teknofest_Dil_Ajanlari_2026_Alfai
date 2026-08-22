# Şartnamenin 4. ve 5. maddelerinde kesin olarak uymamız istenen, 
# hem insanın okuyabileceği hem de makinenin işleyebileceği yapılandırılmış veri modeli.

from pydantic import BaseModel, Field
from typing import List, Union, Optional

class CoordinatePoint(BaseModel):
    x: float = Field(..., ge=0, le=1)
    y: float = Field(..., ge=0, le=1)

class BoundingBox(BaseModel):
    x1: float = Field(..., ge=0, le=1)
    y1: float = Field(..., ge=0, le=1)
    x2: float = Field(..., ge=0, le=1)
    y2: float = Field(..., ge=0, le=1)

class RankedAction(BaseModel):
    rank: int = Field(..., ge=1, le=3)
    label: str = Field(..., description="Operatörün uygulayabileceği aksiyon")
    reason: str = Field(default="", description="Aksiyonun sıralama gerekçesi")

class EvidenceItem(BaseModel):
    time: str = Field(..., description="Kanıtın zaman damgası (Örn: '00:15')")
    seconds: Optional[float] = Field(default=None, description="Kanıtın saniye cinsinden zaman değeri")
    description: str = Field(..., description="Kanıtın Türkçe açıklaması")

class EventItem(BaseModel):
    time: str = Field(..., description="Olayın gerçekleştiği zaman damgası (Örn: '00:15')")
    start_time: Optional[Union[str, float, int]] = Field(default=None, description="Olayın başlangıç zaman damgası")
    end_time: Optional[Union[str, float, int]] = Field(default=None, description="Olayın bitiş zaman damgası")
    # LangGraph Yapılandırma Hatası Çözümü dosyasındaki uyarılara istinaden:
    # Hem düz metin hem de olası sözlük yapılarını desteklemek için Union kullanılmıştır.
    event: Union[str, dict] = Field(..., description="Tespit edilen olayın Türkçe açıklaması veya detay objesi")
    event_type: Optional[str] = Field(default=None, description="Olayın türü (Örn: 'Yangın başlangıcı', 'Patlama', 'İşçi hareketi')")
    zone: Optional[str] = Field(default=None, description="Olayın gerçekleştiği bölge/alan")
    # ==========================================
    # Zamansal Farkındalık ve Kritik An Analizi (Şartname Gereği)
    # ==========================================
    phase: Optional[str] = Field(
        default=None,
        description="Olayın gelişim aşaması: 'başlangıç', 'gelişim' veya 'sonuç'"
    )
    is_critical_moment: Optional[bool] = Field(
        default=False,
        description="Bu an kritik bir olay mı? (anomali, kaza, risk vb.) — True ise vurgulanır"
    )
    risk: Optional[str] = Field(
        default="Düşük",
        description="Bu olayın risk seviyesi (Düşük / Orta / Yüksek / Kritik)"
    )
    confidence: Optional[Union[float, str]] = Field(
        default=None,
        description="Modelin bu olay için güven skoru (0.0–1.0 veya '%85' formatında)"
    )
    evidence: Optional[List[EvidenceItem]] = Field(
        default_factory=list,
        description="Bu olayı destekleyen zamanlı kanıt listesi"
    )
    uncertainties: Optional[List[str]] = Field(
        default_factory=list,
        description="Bu olayla ilgili belirsizlikler veya doğrulama notları"
    )
    center: Optional[CoordinatePoint] = Field(default=None, description="Video karesindeki normalize olay merkezi")
    bbox: Optional[BoundingBox] = Field(default=None, description="Video karesindeki normalize olay kutusu")
    ranked_actions: List[RankedAction] = Field(default_factory=list, description="Olaya özel öncelik sıralı aksiyonlar")

class KpiMetrics(BaseModel):
    ai_processing_status: str = Field(..., description="Yapay zeka işlem durumu (Örn: 'Başarılı / Kritik Müdahale Devrede')")
    confidence_score: float = Field(..., description="Modelin ürettiği ortalama güven skoru (0.0–1.0)")
    is_schema_validated: bool = Field(..., description="Şema doğrulama durumu (True/False)")
    # ==========================================
    # Şartname KPI Metrikleri
    # ==========================================
    detected_events_count: Optional[int] = Field(default=None, description="Tespit edilen toplam olay sayısı (Olay Tespit Doğruluğu göstergesi)")
    critical_events_count: Optional[int] = Field(default=None, description="is_critical_moment=True olan olay sayısı")
    critical_event_capture_rate: Optional[float] = Field(default=None, description="Kritik Olay Yakalama Oranı: kritik_olay / toplam_olay (0.0–1.0)")
    total_processing_seconds: Optional[float] = Field(default=None, description="Toplam pipeline işlem süresi (saniye)")
    action_recommendation_count: Optional[int] = Field(default=None, description="Aksiyon Önerisi Doğruluğu göstergesi: sistemin önerdiği aksiyon sayısı")

class FinalReportSchema(BaseModel):
    summary: str = Field(..., description="Videonun genel operasyonel Türkçe özeti")
    events: List[EventItem] = Field(..., description="Zaman damgalı olayların listesi")
    risk: str = Field(..., description="Risk değerlendirme seviyesi (Düşük, Orta, Yüksek veya Sistem Hatası)")
    
    # ==========================================
    # YENİ EKLENEN KISIM: Geçmiş hafızadan çekilen risk eğilimi
    # ==========================================
    risk_trend: Optional[str] = Field(default="Bilinmiyor", description="Risk değişim eğilimi (Örn: 'ARTAN TREND', 'STABİL')")
    
    actions: List[str] = Field(..., description="Operatöre sunulan acil aksiyon önerileri")
    kpi_metrics: KpiMetrics = Field(..., description="Şartname metrik gereksinimi uyarınca otonom hesaplanan yapay zeka performans verileri")
