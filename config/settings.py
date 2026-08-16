from pydantic import ConfigDict  
from pydantic_settings import BaseSettings
from typing import Dict, Any

class Settings(BaseSettings):
    PROJECT_NAME: str = "TEKNOFEST AI Agent"
    DEBUG: bool = True
    REASONING_MODEL: str = "Llama-3.1-70B"
    VISION_MODEL: str = "Qwen/Qwen3-VL-8B-Instruct"
    VISION_SERVER_URL: str = "http://localhost:8000"
    BRAIN_SERVER_URL: str = "http://localhost:8001"
    VLLM_SERVER_URL: str = "http://localhost:8001"
    CMS_EXPORT_URL: str = "http://localhost:8080/v1/incident/export"
    
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
