from openai import OpenAI

from model_py.config.settings import (
    API_KEY,
    VISION_BASE_URL,
    BRAIN_BASE_URL,
)

vision_client = OpenAI(
    api_key=API_KEY,
    base_url=VISION_BASE_URL,
)

brain_client = OpenAI(
    api_key=API_KEY,
    base_url=BRAIN_BASE_URL,
)
