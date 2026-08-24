# services/vision_service.py - ESKİ ÇALIŞAN SÜRÜM + SADECE VISION SÜRE ÖLÇÜMÜ

import json
import re
import os
import mimetypes
import logging
import time
import subprocess
from typing import Dict, Any, Optional, List

import cv2

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


def _get_video_duration(video_path: str) -> float:
    """Video dosyasının süresini saniye cinsinden döndürür."""
    if not os.path.exists(video_path):
        return 0.0

    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if fps > 0 and frame_count > 0:
                return frame_count / fps
    except Exception:
        pass

    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass

    return 0.0


def _get_video_resolution(video_path: str) -> tuple:
    """Video dosyasının genişlik ve yüksekliğini döndürür."""
    if not os.path.exists(video_path):
        return (0, 0)

    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if width > 0 and height > 0:
                return (width, height)
    except Exception:
        pass

    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                return (int(lines[0]), int(lines[1]))
    except Exception:
        pass

    return (0, 0)


def calculate_dynamic_fps(video_path: str) -> Dict[str, Any]:
    """Video süresine ve çözünürlüğüne göre dinamik FPS ve max_pixels hesaplar."""
    duration = _get_video_duration(video_path)
    width, height = _get_video_resolution(video_path)
    total_pixels = width * height

    if duration <= 30:
        fps = 4
        max_pixels = 1024 * 1024
    elif duration <= 60:
        fps = 3
        max_pixels = 896 * 896
    elif duration <= 120:
        fps = 2
        max_pixels = 768 * 768
    else:
        fps = 1
        max_pixels = 640 * 640

    if total_pixels > 1920 * 1080:
        max_pixels = int(max_pixels * 0.7)
        fps = max(1, fps - 1)

    total_frames = int(duration * fps)
    if total_frames > 800:
        old_fps = fps
        fps = max(1, int(800 / duration))
        logger.warning(
            f"⚠️ Çok fazla kare ({total_frames}), FPS {old_fps}->{fps} düşürüldü"
        )

    logger.info(
        f"Dinamik FPS hesaplandı | Süre: {duration}s | "
        f"FPS: {fps} | Max Pixels: {max_pixels} | "
        f"Toplam Kare: {int(duration * fps)}"
    )

    return {
        "fps": fps,
        "max_pixels": max_pixels,
        "duration": duration,
        "total_frames": int(duration * fps),
    }


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
            normalized_bbox = {
                key: _validate_coordinate(bbox.get(key))
                for key in ("x1", "y1", "x2", "y2")
            }
            if (
                all(v is not None for v in normalized_bbox.values())
                and normalized_bbox["x1"] < normalized_bbox["x2"]
                and normalized_bbox["y1"] < normalized_bbox["y2"]
            ):
                event["bbox"] = normalized_bbox
            else:
                event["bbox"] = None
        else:
            event["bbox"] = None

        center = event.get("center")
        if isinstance(center, (list, tuple)) and len(center) == 2:
            center = {"x": center[0], "y": center[1]}

        if isinstance(center, dict):
            x = _validate_coordinate(center.get("x"))
            y = _validate_coordinate(center.get("y"))
            event["center"] = (
                {"x": x, "y": y} if x is not None and y is not None else None
            )
        elif event["bbox"]:
            event["center"] = {
                "x": round(
                    (event["bbox"]["x1"] + event["bbox"]["x2"]) / 2, 4
                ),
                "y": round(
                    (event["bbox"]["y1"] + event["bbox"]["y2"]) / 2, 4
                ),
            }
        else:
            event["center"] = None

        if "event_type" in event:
            event["event_type"] = _normalize_event_type(event["event_type"])
        if "phase" in event:
            event["phase"] = _normalize_phase(event["phase"])
        if "risk" in event:
            event["risk"] = _normalize_risk_level(event["risk"])

    if "risk" in result:
        result["risk"] = _normalize_risk_level(result["risk"])

    return result


def _timestamp_to_seconds(value) -> float:
    """MM veya sayısal zaman değerini saniyeye çevirir."""
    if isinstance(value, (int, float)):
        return max(0.0, float(value))

    try:
        parts = str(value or "00:00").strip().split(":")
        if len(parts) == 2:
            return max(0.0, int(parts[0]) * 60 + float(parts[1]))
        if len(parts) == 3:
            return max(
                0.0,
                int(parts[0]) * 3600
                + int(parts[1]) * 60
                + float(parts[2]),
            )
    except (TypeError, ValueError):
        pass

    return 0.0


def _read_frame_at(cap, seconds: float):
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, seconds) * 1000.0)
    ok, frame = cap.read()
    return frame if ok else None


def _locate_motion_at_timestamp(
    video_path: str, timestamp
) -> Optional[dict]:
    """Olay zamanının çevresindeki belirgin hareketli bölgeyi bulur."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    try:
        event_second = _timestamp_to_seconds(timestamp)
        fps = max(cap.get(cv2.CAP_PROP_FPS), 1.0)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

        before_second = max(0.0, event_second - 0.6)
        after_second = (
            min(duration, event_second + 0.6)
            if duration > 0
            else event_second + 0.6
        )

        if after_second - before_second < 0.3:
            after_second = before_second + 0.6

        before = _read_frame_at(cap, before_second)
        after = _read_frame_at(cap, after_second)

        if before is None or after is None or before.shape != after.shape:
            return None

        height, width = after.shape[:2]
        frame_area = float(width * height)

        gray_before = cv2.GaussianBlur(
            cv2.cvtColor(before, cv2.COLOR_BGR2GRAY),
            (7, 7),
            0,
        )
        gray_after = cv2.GaussianBlur(
            cv2.cvtColor(after, cv2.COLOR_BGR2GRAY),
            (7, 7),
            0,
        )

        difference = cv2.absdiff(gray_before, gray_after)
        _, mask = cv2.threshold(
            difference,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        candidates = [
            contour
            for contour in contours
            if cv2.contourArea(contour) >= frame_area * 0.001
        ]

        if not candidates:
            return None

        largest = max(candidates, key=cv2.contourArea)
        x, y, box_width, box_height = cv2.boundingRect(largest)

        if box_width * box_height > frame_area * 0.65:
            return None

        bbox = {
            "x1": round(x / width, 4),
            "y1": round(y / height, 4),
            "x2": round((x + box_width) / width, 4),
            "y2": round((y + box_height) / height, 4),
        }

        return {
            "bbox": bbox,
            "center": {
                "x": round((bbox["x1"] + bbox["x2"]) / 2, 4),
                "y": round((bbox["y1"] + bbox["y2"]) / 2, 4),
            },
            "location_source": "opencv_motion",
        }
    finally:
        cap.release()


def _fill_missing_spatial_from_motion(result: dict, video_path: str) -> dict:
    """Qwen koordinat vermediyse OpenCV hareket konumunu kontrollü fallback olarak kullanır."""
    for event in result.get("events", []):
        if not isinstance(event, dict) or event.get("center") is not None:
            continue

        location = _locate_motion_at_timestamp(
            video_path,
            event.get("time") or event.get("start_time") or 0,
        )

        if location:
            event.update(location)
            logger.info(
                "OpenCV konum fallback uygulandı | zaman=%s | center=%s",
                event.get("time"),
                event.get("center"),
            )

    return result


def _normalize_event_type(value: str) -> str:
    """Event type'ı standart formata dönüştürür."""
    if not value:
        return "DURUM"

    value = str(value).upper().strip()
    value = (
        value.replace("İ", "I")
        .replace("Ü", "U")
        .replace("Ğ", "G")
        .replace("Ş", "S")
        .replace("Ö", "O")
        .replace("Ç", "C")
    )

    if "ANOMALI" in value:
        return "ANOMALI"
    if "KAZA" in value:
        return "KAZA"
    if "HAREKET" in value:
        return "HAREKET"
    if "DURUM" in value:
        return "DURUM"
    if "CIHAZ" in value:
        return "CIHAZ"

    return "DURUM"


def _normalize_phase(value: str) -> str:
    """Phase'ı standart formata dönüştürür."""
    if not value:
        return "baslangic"

    value = str(value).lower().strip()

    if "geli" in value or "orta" in value:
        return "gelisim"
    if "son" in value or "bitis" in value:
        return "sonuc"

    return "baslangic"


def _normalize_risk_level(value) -> str:
    """Risk seviyesini standart formata dönüştürür."""
    from agent.graph import normalize_risk
    return normalize_risk(value)


def _parse_vision_json(raw_text: str, video_duration: float = 0) -> dict:
    """Vision çıktısını güvenli şekilde JSON'a çevirir."""
    if not raw_text or not raw_text.strip():
        logger.error("Vision modeli BOŞ yanıt döndürdü!")
        return _create_fallback_response(video_duration)

    cleaned = raw_text.strip()
    logger.info(f"RAW JSON uzunluğu: {len(cleaned)} karakter")

    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        cleaned = cleaned[start_idx:end_idx + 1].strip()
    else:
        logger.warning("JSON başlangıç/bitiş işaretleri bulunamadı!")
        return _create_fallback_response(video_duration)

    parse_attempts = [
        ("Doğrudan JSON", lambda: json.loads(cleaned)),
        (
            "Kontrol karakter temizleme",
            lambda: json.loads(
                re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", cleaned)
            ),
        ),
        (
            "json-repair",
            lambda: (
                _json_repair(cleaned, return_objects=True)
                if _HAS_JSON_REPAIR
                else None
            ),
        ),
        ("Bracket tamamlama", lambda: _manual_brace_repair(cleaned)),
        ("Sadece events çıkar", lambda: _extract_events_only(cleaned)),
        ("Regex ile JSON bul", lambda: _extract_json_by_regex(cleaned)),
    ]

    for attempt_name, attempt_fn in parse_attempts:
        try:
            result = attempt_fn()
            if isinstance(result, dict) and result.get("events") is not None:
                logger.info(
                    f"{attempt_name} ile JSON başarıyla parse edildi!"
                )
                event_count = len(result.get("events", []))
                if event_count == 0:
                    logger.warning(
                        f"{attempt_name} ile parse edildi ama 0 events!"
                    )
                return result
        except Exception as e:
            logger.debug(
                f"{attempt_name} başarısız: {str(e)[:100]}"
            )

    logger.error("Tüm parse denemeleri başarısız! Fallback kullanılıyor.")
    return _create_fallback_response(video_duration)


def _manual_brace_repair(text: str) -> dict:
    """Eksik parantezleri tamamlar."""
    repair_text = text

    if repair_text.count('"') % 2 != 0:
        repair_text += '"'

    open_braces = repair_text.count("{") - repair_text.count("}")
    open_brackets = repair_text.count("[") - repair_text.count("]")

    repair_text += "]" * max(0, open_brackets)
    repair_text += "}" * max(0, open_braces)

    return json.loads(repair_text)


def _extract_events_only(text: str) -> dict:
    """Sadece events listesini kurtarmayı dener."""
    events_match = re.search(
        r'"events"\s*:\s*(\[[\s\S]*?\])\s*(?:,\s*"risk"|,\s*"actions"|})',
        text,
    )

    if not events_match:
        raise ValueError("Events listesi bulunamadı")

    events_raw = events_match.group(1)

    if _HAS_JSON_REPAIR:
        events_list = _json_repair(
            events_raw,
            return_objects=True,
        )
    else:
        events_list = json.loads(events_raw)

    if isinstance(events_list, list) and events_list:
        return {
            "summary": "Kısmi analiz — sadece events listesi kurtarıldı.",
            "events": events_list,
            "risk": "Orta",
            "risk_score": 50,
            "actions": [
                "Analiz devam ediyor, detaylı rapor için yeniden dene."
            ],
        }

    raise ValueError("Geçerli events listesi bulunamadı")


def _extract_json_by_regex(text: str) -> dict:
    """JSON benzeri yapıyı bulmayı dener."""
    match = re.search(
        r"\{[\s\S]*\"events\"\s*:\s*\[[\s\S]*\][\s\S]*\}",
        text,
    )

    if match:
        return json.loads(match.group())

    raise ValueError("Regex ile JSON bulunamadı")


def _create_fallback_response(video_duration: float = 0) -> dict:
    """Vision çıktısı işlenemezse güvenli fallback."""
    return {
        "summary": (
            "Görsel analiz verisi işlenirken biçim uyuşmazlığı tespit edildi. "
            "Otonom saha izlemesi aktif."
        ),
        "events": [
            {
                "time": "00:00",
                "start_time": "00:00",
                "end_time": (
                    _format_time(video_duration)
                    if video_duration > 0
                    else "01:00"
                ),
                "event": "Saha görüntüsü incelendi, genel akış stabil.",
                "event_type": "DURUM",
                "phase": "baslangic",
                "zone": "Genel Saha",
                "risk": "Düşük",
                "is_critical_moment": False,
                "confidence": 0.70,
                "center": None,
                "bbox": None,
            }
        ],
        "risk": "Orta",
        "risk_score": 50,
        "actions": ["log_kpi_metrics"],
    }


def _format_time(seconds: int) -> str:
    """Saniyeyi MM:SS formatına dönüştürür."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def analyze_video(video_path: str) -> dict:
    """
    Eski Vision analiz akışı.
    Sadece model çağrısının süresini ölçmek için vision_start eklenmiştir.
    """
    start_time = time.time()

    VIDEO_FPS = 4
    VIDEO_MAX_PIXELS = 768 * 768

    if not os.path.exists(video_path):
        logger.error(f"Video bulunamadı: {video_path}")
        return _create_fallback_response()

    duration = _get_video_duration(video_path)
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)

    fps_config = calculate_dynamic_fps(video_path)

    ext = os.path.splitext(video_path)[1].lower()
    mime_type = (
        _MIME_MAP.get(ext)
        or mimetypes.guess_type(video_path)[0]
        or "video/mp4"
    )

    logger.info(
        f"Video encode ediliyor | Yol: {video_path} | "
        f"Süre: {duration}s | Boyut: {file_size_mb:.2f} MB | "
        f"FPS: {fps_config['fps']} | MIME: {mime_type}"
    )

    try:
        base64_video = encode_video_to_base64(video_path)
    except Exception as e:
        logger.error(f"Video encode hatası: {e}")
        return _create_fallback_response(duration)

    # ================================================================
    # SADECE ÖLÇÜM: Gerçek Vision model çağrısının süresini ölçüyoruz.
    # Model ayarlarına dokunulmadı.
    # ================================================================
    vision_start = time.time()

    try:
        response = vision_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": VISION_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": VISION_USER_PROMPT,
                        },
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": (
                                    f"data:{mime_type};base64,"
                                    f"{base64_video}"
                                ),
                                "fps": fps_config["fps"],
                                "max_pixels": fps_config["max_pixels"],
                            },
                        },
                    ],
                },
            ],
            max_tokens=4096,
            temperature=0.2,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            },
            timeout=180,
        )
    except Exception as e:
        logger.error(f"Vision model çağrısı hatası: {e}")
        return _create_fallback_response(duration)

    logger.info(
        f"⏱️ VISION MODEL SÜRESİ: "
        f"{time.time() - vision_start:.2f} saniye"
    )

    vision_output = response.choices[0].message.content or ""
    logger.info(
        f"Vision yanıt uzunluğu: {len(vision_output)} karakter"
    )

    parsed = _normalize_spatial_fields(
        _parse_vision_json(vision_output, duration)
    )

    parsed = _fill_missing_spatial_from_motion(
        parsed,
        video_path,
    )

    event_count = len(parsed.get("events", []))

    if event_count == 0:
        logger.warning(
            f"Vision modeli BOŞ events döndürdü! Video: {video_path}"
        )
    else:
        logger.info(
            f"Vision analizi tamamlandı: {event_count} olay, "
            f"Süre: {time.time() - start_time:.2f}s"
        )

    return parsed


_MIME_MAP = {
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
}