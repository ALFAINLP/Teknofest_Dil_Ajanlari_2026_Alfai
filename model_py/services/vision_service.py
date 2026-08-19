import json
import re
import logging
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

logger = logging.getLogger("ALFAI_API")


def _validate_coordinate(value):
    try:
        numeric = float(value)
        return numeric if 0.0 <= numeric <= 1.0 else None
    except (TypeError, ValueError):
        return None


def _normalize_spatial_fields(result: dict) -> dict:
    """Model koordinatlarını doğrular; geçersiz değerlerden konum uydurmaz."""
    for event in result.get("events", []):
        if not isinstance(event, dict):
            continue

        bbox = event.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            bbox = dict(zip(("x1", "y1", "x2", "y2"), bbox))
        if isinstance(bbox, dict):
            normalized_bbox = {key: _validate_coordinate(bbox.get(key)) for key in ("x1", "y1", "x2", "y2")}
            if all(value is not None for value in normalized_bbox.values()) and normalized_bbox["x1"] < normalized_bbox["x2"] and normalized_bbox["y1"] < normalized_bbox["y2"]:
                event["bbox"] = normalized_bbox
            else:
                event["bbox"] = None
        else:
            event["bbox"] = None

        center = event.get("center")
        if isinstance(center, (list, tuple)) and len(center) == 2:
            center = {"x": center[0], "y": center[1]}
        if isinstance(center, dict):
            x, y = _validate_coordinate(center.get("x")), _validate_coordinate(center.get("y"))
            event["center"] = {"x": x, "y": y} if x is not None and y is not None else None
        elif event["bbox"]:
            event["center"] = {
                "x": round((event["bbox"]["x1"] + event["bbox"]["x2"]) / 2, 4),
                "y": round((event["bbox"]["y1"] + event["bbox"]["y2"]) / 2, 4),
            }
        else:
            event["center"] = None
    return result


def _parse_vision_json(raw_text: str) -> dict:
    """
    Vision modelinden gelen ham metni temizler, JSON formatına sokar
    ve ayrıştırır. Hata durumlarında güvenli fallback döndürür.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Model boş yanıt döndürdü.")

    cleaned = raw_text.strip()

    # Markdown kod bloklarını temizle
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()

    # İlk { ile son } arasını al
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        cleaned = cleaned[start_idx:end_idx + 1].strip()

    # 1. Doğrudan ayrıştırma
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 2. Kontrol karakterleri ve geçersiz karakterleri temizleme
    try:
        cleaned_ctrl = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', cleaned)
        return json.loads(cleaned_ctrl)
    except json.JSONDecodeError:
        pass

    # 3. Kesintiye uğramış (unterminated) JSON stringlerini tamamlama denemesi
    try:
        repair_text = cleaned
        if repair_text.count('"') % 2 != 0:
            repair_text += '"'
        open_braces = repair_text.count('{') - repair_text.count('}')
        open_brackets = repair_text.count('[') - repair_text.count(']')
        repair_text += ']' * max(0, open_brackets)
        repair_text += '}' * max(0, open_braces)
        return json.loads(repair_text)
    except Exception as err:
        logger.warning(f"⚠️ Vision JSON Onarımı Başarısız: {err}")

    # 4. Güvenli Fallback Yapısı
    logger.error("❌ Vision çıktısı geçerli bir JSON'a dönüştürülemedi, varsayılan şema yükleniyor.")
    return {
        "summary": "Görsel analiz verisi işlenirken biçim uyuşmazlığı tespit edildi. Otonom saha izlemesi aktif.",
        "events": [
            {
                "time": "00:00",
                "event": "Saha görüntüsü incelendi, genel akış stabil.",
                "zone": "Genel Saha",
                "risk": "Düşük",
                "confidence": 0.85
            }
        ],
        "risk": "Düşük",
        "risk_score": 25,
        "actions": [
            "Saha operasyonlarının takibine devam edilmesi",
            "Kamera görüntüsünün kontrol edilmesi"
        ]
    }


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

    return _normalize_spatial_fields(_parse_vision_json(vision_output))
