from model_py.config.settings import VISION_BASE_URL, BRAIN_BASE_URL
from openai import OpenAI

# Evren API Yapılandırması
EVREN_BASE_URL: str = "https://evren-llmapi.ssyz.org.tr/v1"
EVREN_API_KEY: str = "sk-evren-team31-44c78fc209d8d741cbb8b180c08ce452"


vision_client = OpenAI(
    api_key=EVREN_API_KEY,
    base_url=EVREN_BASE_URL,
)

brain_client = OpenAI(
    api_key="dummy",
    base_url=BRAIN_BASE_URL,
)