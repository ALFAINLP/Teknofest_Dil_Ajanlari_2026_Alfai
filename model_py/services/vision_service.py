import json
from model_py.services.client import vision_client
from model_py.utils.video_utils import encode_video_to_base64
from model_py.config.settings import (
    VISION_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
)
from model_py.prompts.vision_prompt import (
    VISION_SYSTEM_PROMPT,
    VISION_USER_PROMPT,
)


def analyze_video(video_path: str):
    """
    Vision modelini kullanarak videoyu analiz eder
    ve JSON çıktısını döndürür.
    """

    base64_video = encode_video_to_base64(video_path)

    response = vision_client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "system",
                "content": VISION_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": VISION_USER_PROMPT
                    },
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": f"data:video/mp4;base64,{base64_video}"
                        }
                    }
                ]
            }
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        response_format={"type": "json_object"},
    )

    vision_output = response.choices[0].message.content

    return json.loads(vision_output)
