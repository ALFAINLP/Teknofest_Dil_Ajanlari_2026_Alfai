from pydantic import ConfigDict  
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "TEKNOFEST AI Agent"
    DEBUG: bool = True

    # API Yapılandırması (.env dosyasından dinamik olarak okunacak)
    EVREN_BASE_URL: Optional[str] = None
    EVREN_API_KEY: Optional[str] = None

    # Model Tanımları (Brain için llm-fast, Vision için 32B model)
    REASONING_MODEL: str = "llm-large"
    VISION_MODEL: str = "Qwen/Qwen3-VL-32B-Instruct"

    # CMS URL (.env'den okunacak)
    CMS_EXPORT_URL: Optional[str] = None
    
    TEMPERATURE: float = 0.0
    MAX_TOKENS: int = 2048

    model_config = ConfigDict(env_file=".env")

settings = Settings()