from pydantic import ConfigDict  
from pydantic_settings import BaseSettings
from typing import Dict, Any

class Settings(BaseSettings):
    PROJECT_NAME: str = "TEKNOFEST AI Agent"
    DEBUG: bool = True

    # Evren API Yapılandırması (Baştaki boşluk temizlendi)
    EVREN_BASE_URL: str = "https://evren-llmapi.ssyz.org.tr/v1"
    EVREN_API_KEY: str = "sk-evren-team31-44c78fc209d8d741cbb8b180c08ce452"

    # Model Tanımları (Brain için llm-fast, Vision için 32B model)
    REASONING_MODEL: str = "llm-large"
    VISION_MODEL: str = "Qwen/Qwen3-VL-32B-Instruct"

    CMS_EXPORT_URL: str = "http://localhost:8080/v1/incident/export"
    
    TEMPERATURE: float = 0.0
    MAX_TOKENS: int = 2048

    MOCK_VIDEO_TIMELINE: Dict[int, Dict[str, Any]] = {
        0: {
            "frame": "frame_0.jpg", 
            "description": "Saha operasyonu sakin, personel normal seyrinde çalışıyor.", 
            "has_risk": False
        },
        15: {
            "frame": "frame_15.jpg", 
            "description": "Forklift hızlı bir şekilde virajı alıyor, devrilme ve çarpma riski yüksek.", 
            "has_risk": True, 
            "risk_type": "İş Güvenliği İhlali"
        },
        30: {
            "frame": "frame_30.jpg", 
            "description": "Forklift devrilmiş durumda, operatör yerde hareketsiz yatıyor.", 
            "has_risk": True, 
            "risk_type": "Acil Sağlık Durumu / İş Kazası"
        },
        45: {
            "frame": "frame_45.jpg", 
            "description": "Çevre çalışanlar ilk yardım için olay yerine doğru intikal ediyor.", 
            "has_risk": False
        }
    }

    model_config = ConfigDict(env_file=".env")

settings = Settings()