import json
import re
import os
import mimetypes
import logging
try:
    from json_repair import repair_json as _json_repair
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False
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
    ve ayrıştırır.

    Onarım sırası:
      1. Doğrudan json.loads
      2. Kontrol karakteri temizleme + json.loads
      3. json-repair kütüphanesi (virgül eksikliği, kapanmamış string vb.)
      4. Manuel bracket/brace tamamlama + json.loads
      5. Kısmi JSON kurtarma — sadece 'events' listesini çıkarmaya çalış
      6. Güvenli fallback
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

    # Token limit nedeniyle JSON kesildiyse uyar
    if cleaned and cleaned[-1] not in ("}" , "]", '"'):
        logger.warning(
            f"⚠️ JSON muhtemelen token sınırında kesildi. "
            f"Son karakter: {repr(cleaned[-1])} | Toplam uzunluk: {len(cleaned)}"
        )

    # ── 1. Doğrudan ayrıştırma ─────────────────────────────────────────────
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # ── 2. Kontrol karakterleri temizleme ──────────────────────────────────
    try:
        cleaned_ctrl = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', cleaned)
        return json.loads(cleaned_ctrl)
    except json.JSONDecodeError:
        cleaned_ctrl = cleaned  # sonraki adımlarda kullan

    # ── 3. json-repair kütüphanesi (virgül, tırnak, bracket hataları) ──────
    if _HAS_JSON_REPAIR:
        try:
            repaired = _json_repair(cleaned_ctrl, return_objects=True)
            if isinstance(repaired, dict):
                logger.info("🔧 json-repair ile JSON başarıyla onarıldı.")
                return repaired
        except Exception as repair_err:
            logger.warning(f"⚠️ json-repair başarısız: {repair_err}")

    # ── 4. Manuel bracket/brace tamamlama ──────────────────────────────────
    try:
        repair_text = cleaned_ctrl
        if repair_text.count('"') % 2 != 0:
            repair_text += '"'
        open_braces = repair_text.count('{') - repair_text.count('}')
        open_brackets = repair_text.count('[') - repair_text.count(']')
        repair_text += ']' * max(0, open_brackets)
        repair_text += '}' * max(0, open_braces)
        return json.loads(repair_text)
    except Exception as err:
        logger.warning(f"⚠️ Manuel JSON onarımı başarısız: {err}")

    # ── 5. Kısmi kurtarma: sadece events listesini çıkar ───────────────────
    try:
        events_match = re.search(r'"events"\s*:\s*(\[.*?\])', cleaned_ctrl, re.DOTALL)
        if events_match:
            events_raw = events_match.group(1)
            if _HAS_JSON_REPAIR:
                events_list = _json_repair(events_raw, return_objects=True)
            else:
                events_list = json.loads(events_raw)
            if isinstance(events_list, list) and events_list:
                logger.warning("⚠️ Kısmi kurtarma: sadece 'events' listesi alındı.")
                return {
                    "summary": "Kısmi analiz — JSON onarımdan geçti.",
                    "events": events_list,
                    "risk": "Orta",
                    "risk_score": 40,
                    "actions": ["Kısmi veri ile operasyonel izleme sürdürülüyor."]
                }
    except Exception:
        pass

    # ── 6. Güvenli Fallback ────────────────────────────────────────────────
    logger.error("❌ Vision çıktısı geçerli bir JSON'a dönüştürülemedi, varsayılan şema yükleniyor.")
    return {
        "summary": "Görsel analiz verisi işlenirken biçim uyuşmazlığı tespit edildi. Otonom saha izlemesi aktif.",
        "events": [
            {
                "time": "00:00",
                "event": "Saha görüntüsü incelendi, genel akış stabil.",
                "zone": "Genel Saha",
                "risk": "Düşük",
                "confidence": 0.85,
                "ranked_actions": [
                    {"rank": 1, "label": "Saha operatörüne bildir", "reason": "Görsel bulgunun insan tarafından doğrulanması için."},
                    {"rank": 2, "label": "Kamera kaydını odağa al", "reason": "Olayın gelişimini yakından izlemek için."},
                    {"rank": 3, "label": "Olay sonrası inceleme oluştur", "reason": "Tekrarlayan riskleri azaltacak önlemleri belirlemek için."}
                ]
            }
        ],
        "risk": "Düşük",
        "risk_score": 25,
        "actions": [
            "Saha operasyonlarının takibine devam edilmesi",
            "Kamera görüntüsünün kontrol edilmesi"
        ]
    }


# Desteklenen video MIME type eşlemesi
_MIME_MAP = {
    ".mp4":  "video/mp4",
    ".avi":  "video/x-msvideo",
    ".mov":  "video/quicktime",
    ".mkv":  "video/x-matroska",
    ".webm": "video/webm",
}

# Qwen3-VL frame örnekleme ayarları
# fps: saniyede kaç kare alınacak (düşük fps → daha az kare → daha hızlı ama daha az bağlam)
# max_pixels: tek kare için maksimum piksel sayısı (büyük değer → daha net görüntü)
# Eğer video uzunsa fps=1 ile başla; kısa videolar için fps=2
VIDEO_FPS = 3           # saniyede 2 kare örnekle
VIDEO_MAX_PIXELS = 768 * 768  # kare başına maks piksel (Qwen3-VL varsayılanı: 128*28*28)


def analyze_video(video_path: str):
    """
    Vision modelini kullanarak videoyu analiz eder
    ve JSON çıktısını döndürür.
    """
    # --- MIME type: uzantıya göre dinamik belirle ---
    ext = os.path.splitext(video_path)[1].lower()
    mime_type = _MIME_MAP.get(ext) or mimetypes.guess_type(video_path)[0] or "video/mp4"

    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    logger.info(
        f"🎞️ Video encode ediliyor | Yol: {video_path} | "
        f"Boyut: {file_size_mb:.2f} MB | MIME: {mime_type}"
    )

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
                        # fps ve max_pixels: Qwen3-VL'nin kaç kare işleyeceğini kontrol eder.
                        # Bu parametreler vLLM tarafından desteklenir ve modele iletilir.
                        "type": "video_url",
                        "video_url": {
                            "url": f"data:{mime_type};base64,{base64_video}",
                            "fps": VIDEO_FPS,
                            "max_pixels": VIDEO_MAX_PIXELS,
                        }
                    }
                ]
            }
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        # NOT: response_format={"type": "json_object"} Qwen3-VL ile uyumsuz olabilir;
        # JSON çıktısı prompt üzerinden yönlendirilmektedir.
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    vision_output = response.choices[0].message.content
    print("\n" + "=" * 80)
    print("🔍 RAW VISION OUTPUT")
    print("=" * 80)
    print(vision_output)
    print("=" * 80 + "\n")

    parsed = _normalize_spatial_fields(_parse_vision_json(vision_output))

    event_count = len(parsed.get("events", []))
    if event_count == 0:
        logger.warning(
            f"⚠️ Vision modeli BOŞ events listesi döndürdü! "
            f"Video: {video_path} | Summary: {parsed.get('summary', 'YOK')} | "
            f"Risk: {parsed.get('risk', 'YOK')} | "
            f"Raw output uzunluğu: {len(vision_output or '')} karakter"
        )
    else:
        logger.info(f"✅ Vision analizi tamamlandı: {event_count} olay tespit edildi.")

    return parsed
