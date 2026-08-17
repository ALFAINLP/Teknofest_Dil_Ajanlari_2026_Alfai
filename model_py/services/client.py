from model_py.config.settings import VISION_BASE_URL, BRAIN_BASE_URL
from openai import OpenAI

vision_client = OpenAI(
    api_key="dummy",
    base_url=VISION_BASE_URL,
)

brain_client = OpenAI(
    api_key="dummy",
    base_url=BRAIN_BASE_URL,
)