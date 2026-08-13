# Şartnamenin 4. ve 5. maddelerinde kesin olarak uymamız istenen, 
# hem insanın okuyabileceği hem de makinenin işleyebileceği yapılandırılmış veri modeli.

from pydantic import BaseModel, Field
from typing import List, Union, Optional  # <-- Optional eklendi

class EventItem(BaseModel):
    time: str = Field(..., description="Olayın gerçekleştiği zaman damgası (Örn: '00:15')")
    # LangGraph Yapılandırma Hatası Çözümü dosyasındaki uyarılara istinaden:
    # Hem düz metin hem de olası sözlük yapılarını desteklemek için Union kullanılmıştır.
    event: Union[str, dict] = Field(..., description="Tespit edilen olayın Türkçe açıklaması veya detay objesi")

class KpiMetrics(BaseModel):
    ai_processing_status: str = Field(..., description="Yapay zeka işlem durumu (Örn: 'Başarılı / Kritik Müdahale Devrede')")
    confidence_score: float = Field(..., description="Modelin ürettiği doğruluk/güven skoru (Örn: 0.96)")
    is_schema_validated: bool = Field(..., description="Şema doğrulama durumu (True/False)")

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
