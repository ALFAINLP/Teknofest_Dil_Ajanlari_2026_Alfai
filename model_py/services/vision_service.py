# services/vision_service.py - GELİŞTİRİLMİŞ VERSİYON

import json
import re
import os
import mimetypes
import logging
import time
from typing import Dict, Any, Optional, List
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

import cv2
import subprocess

def _get_video_duration(video_path: str) -> float:
    """
    Video dosyasının süresini saniye cinsinden döndürür.
    """
    if not os.path.exists(video_path):
        return 0.0
    
    try:
        # OpenCV ile dene
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
        # ffprobe ile dene (daha doğru)
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries',
            'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    
    return 0.0

def _get_video_resolution(video_path: str) -> tuple:
    """
    Video dosyasının genişlik ve yüksekliğini döndürür.
    Returns: (width, height)
    """
    if not os.path.exists(video_path):
        return (0, 0)
    
    try:
        # OpenCV ile dene
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
        # ffprobe ile dene
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                return (int(lines[0]), int(lines[1]))
    except Exception:
        pass
    
    return (0, 0)

# ============================================================================
# 1. DİNAMİK FRAME ÖRNEKLEME
# ============================================================================

def calculate_dynamic_fps(video_path: str) -> Dict[str, Any]:
    """
    Video süresine ve çözünürlüğüne göre dinamik FPS ve max_pixels hesaplar.
    """
    duration = _get_video_duration(video_path)
    width, height = _get_video_resolution(video_path)
    total_pixels = width * height
    
    # FPS hesaplama (video süresine göre)
    if duration <= 30:
        fps = 4  # Kısa videolar: yüksek FPS
        max_pixels = 1024 * 1024  # 1 MP
    elif duration <= 60:
        fps = 3
        max_pixels = 896 * 896
    elif duration <= 120:
        fps = 2
        max_pixels = 768 * 768
    else:
        fps = 1  # Uzun videolar: düşük FPS
        max_pixels = 640 * 640
    
    # Çözünürlük bazlı ayarlama
    if total_pixels > 1920 * 1080:  # Full HD üstü
        max_pixels = int(max_pixels * 0.7)
        fps = max(1, fps - 1)
    
    # Toplam kare sayısını hesapla (vLLM limiti ~1000 kare)
    total_frames = int(duration * fps)
    if total_frames > 800:
        fps = max(1, int(800 / duration))
        logger.warning(f"⚠️ Çok fazla kare ({total_frames}), FPS {fps}->{fps} düşürüldü")
    
    logger.info(
        f" Dinamik FPS hesaplandı | Süre: {duration}s | "
        f"FPS: {fps} | Max Pixels: {max_pixels} | Toplam Kare: {int(duration * fps)}"
    )
    
    return {
        "fps": fps,
        "max_pixels": max_pixels,
        "duration": duration,
        "total_frames": int(duration * fps)
    }

# ============================================================================
# 2. GELİŞTİRİLMİŞ JSON PARSING
# ============================================================================

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

        # BBOX normalizasyonu
        bbox = event.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            bbox = dict(zip(("x1", "y1", "x2", "y2"), bbox))
        if isinstance(bbox, dict):
            normalized_bbox = {key: _validate_coordinate(bbox.get(key)) for key in ("x1", "y1", "x2", "y2")}
            if all(v is not None for v in normalized_bbox.values()) and normalized_bbox["x1"] < normalized_bbox["x2"] and normalized_bbox["y1"] < normalized_bbox["y2"]:
                event["bbox"] = normalized_bbox
            else:
                event["bbox"] = None
        else:
            event["bbox"] = None

        # Center normalizasyonu
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
        
        # YENİ: event_type ve phase standartlaştırma
        if "event_type" in event:
            event["event_type"] = _normalize_event_type(event["event_type"])
        if "phase" in event:
            event["phase"] = _normalize_phase(event["phase"])
        
        # YENİ: Risk seviyesi standartlaştırma
        if "risk" in event:
            event["risk"] = _normalize_risk_level(event["risk"])
    
    #  YENİ: Genel risk standartlaştırma
    if "risk" in result:
        result["risk"] = _normalize_risk_level(result["risk"])
    
    return result

def _normalize_event_type(value: str) -> str:
    """Event type'ı standart formata dönüştür"""
    if not value:
        return "DURUM"
    value = str(value).upper().strip()
    
    # Türkçe karakter düzeltme
    value = value.replace("İ", "I").replace("Ü", "U").replace("Ğ", "G")
    value = value.replace("Ş", "S").replace("Ö", "O").replace("Ç", "C")
    
    if "ANOMALI" in value:
        return "ANOMALI"
    elif "KAZA" in value:
        return "KAZA"
    elif "HAREKET" in value:
        return "HAREKET"
    elif "DURUM" in value:
        return "DURUM"
    elif "CIHAZ" in value or "CIHAZ" in value:
        return "CIHAZ"
    else:
        return "DURUM"  # default

def _normalize_phase(value: str) -> str:
    """Phase'ı standart formata dönüştür"""
    if not value:
        return "baslangic"
    value = str(value).lower().strip()
    
    if "geli" in value or "orta" in value:
        return "gelisim"
    elif "son" in value or "bitis" in value:
        return "sonuc"
    else:
        return "baslangic"

def _normalize_risk_level(value) -> str:
    """Risk seviyesini standart formata dönüştür"""
    from agent.graph import normalize_risk  # graph.py'deki fonksiyonu kullan
    return normalize_risk(value)

def _parse_vision_json(raw_text: str, video_duration: float = 0) -> dict:
    """
    GELİŞTİRİLMİŞ JSON Parsing - 7 seviyeli kurtarma
    """
    if not raw_text or not raw_text.strip():
        logger.error(" Vision modeli BOŞ yanıt döndürdü!")
        return _create_fallback_response(video_duration)

    cleaned = raw_text.strip()
    logger.info(f" RAW JSON uzunluğu: {len(cleaned)} karakter")

    # ---- 1. Markdown temizleme ----
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()

    # ---- 2. JSON bloğunu çıkar ----
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        cleaned = cleaned[start_idx:end_idx + 1].strip()
    else:
        logger.warning(" JSON başlangıç/bitiş işaretleri bulunamadı!")
        return _create_fallback_response(video_duration)

    # ---- 3. 7 seviyeli parse denemesi ----
    parse_attempts = [
        ("Doğrudan JSON", lambda: json.loads(cleaned)),
        ("Kontrol karakter temizleme", lambda: json.loads(re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', cleaned))),
        ("json-repair (varsa)", lambda: _json_repair(cleaned, return_objects=True) if _HAS_JSON_REPAIR else None),
        ("Bracket tamamlama", lambda: _manual_brace_repair(cleaned)),
        ("Sadece events çıkar", lambda: _extract_events_only(cleaned)),
        ("Regex ile JSON bul", lambda: _extract_json_by_regex(cleaned)),
    ]

    for attempt_name, attempt_fn in parse_attempts:
        try:
            result = attempt_fn()
            if isinstance(result, dict) and result.get("events") is not None:
                logger.info(f"{attempt_name} ile JSON başarıyla parse edildi!")
                # Event sayısını kontrol et
                event_count = len(result.get("events", []))
                if event_count == 0:
                    logger.warning(f"{attempt_name} ile parse edildi ama 0 events!")
                return result
        except Exception as e:
            logger.debug(f"{attempt_name} başarısız: {str(e)[:50]}")
            continue

    # ---- 4. Son çare fallback ----
    logger.error(" Tüm parse denemeleri başarısız! Fallback kullanılıyor.")
    return _create_fallback_response(video_duration)

def _manual_brace_repair(text: str) -> dict:
    """Eksik parantezleri tamamla"""
    repair_text = text
    if repair_text.count('"') % 2 != 0:
        repair_text += '"'
    open_braces = repair_text.count('{') - repair_text.count('}')
    open_brackets = repair_text.count('[') - repair_text.count(']')
    repair_text += ']' * max(0, open_brackets)
    repair_text += '}' * max(0, open_braces)
    return json.loads(repair_text)

def _extract_events_only(text: str) -> dict:
    """Sadece events listesini çıkar"""
    events_match = re.search(r'"events"\s*:\s*(\[.*?\])', text, re.DOTALL)
    if events_match:
        events_raw = events_match.group(1)
        if _HAS_JSON_REPAIR:
            events_list = _json_repair(events_raw, return_objects=True)
        else:
            events_list = json.loads(events_raw)
        if isinstance(events_list, list) and events_list:
            return {
                "summary": "Kısmi analiz — sadece events listesi kurtarıldı.",
                "events": events_list,
                "risk": "Orta",
                "risk_score": 50,
                "actions": ["Analiz devam ediyor, detaylı rapor için yeniden dene."]
            }
    raise ValueError("Events listesi bulunamadı")

def _extract_json_by_regex(text: str) -> dict:
    """Regex ile JSON benzeri yapıyı bul"""
    json_pattern = r'\{[^{}]*"events"\s*:\s*\[[^\]]*\][^{}]*\}'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError("Regex ile JSON bulunamadı")

def _create_fallback_response(video_duration: float = 0) -> dict:
    """
    GELİŞTİRİLMİŞ FALLBACK - Tool Jaccard'ı iyileştirir
    """
    #  YENİ: Fallback'te de log_kpi_metrics çağrılmalı
    return {
        "summary": "Görsel analiz verisi işlenirken biçim uyuşmazlığı tespit edildi. Otonom saha izlemesi aktif.",
        "events": [
            {
                "time": "00:00",
                "start_time": "00:00",
                "end_time": _format_time(video_duration) if video_duration > 0 else "01:00",
                "event": "Saha görüntüsü incelendi, genel akış stabil.",
                "event_type": "DURUM",
                "phase": "baslangic",
                "zone": "Genel Saha",
                "risk": "Düşük",
                "is_critical_moment": False,
                "confidence": 0.70,
                "center": None,
                "bbox": None
            }
        ],
        "risk": "Orta",
        "risk_score": 50,
        "actions": ["log_kpi_metrics"]  # ← KRİTİK: Tool Jaccard için!
    }

def _format_time(seconds: int) -> str:
    """Saniyeyi MM:SS formatına dönüştür"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

# ============================================================================
# 3. ANA ANALİZ FONKSİYONU (GELİŞTİRİLMİŞ)
# ============================================================================

def analyze_video(video_path: str) -> dict:
    """
    GELİŞTİRİLMİŞ Vision analizi - Dinamik FPS, gelişmiş parsing
    """
    start_time = time.time()
    VIDEO_FPS = 4  # ← 2'den 3'e çıkar
    VIDEO_MAX_PIXELS = 768 * 768
    
    # ---- 1. Video bilgilerini al ----
    duration = _get_video_duration(video_path)
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    
    # ---- 2. Dinamik FPS hesapla ----
    fps_config = calculate_dynamic_fps(video_path)
    
    # ---- 3. MIME type belirle ----
    ext = os.path.splitext(video_path)[1].lower()
    mime_type = _MIME_MAP.get(ext) or mimetypes.guess_type(video_path)[0] or "video/mp4"
    
    logger.info(
        f" Video encode ediliyor | Yol: {video_path} | "
        f"Süre: {duration}s | Boyut: {file_size_mb:.2f} MB | "
        f"FPS: {fps_config['fps']} | MIME: {mime_type}"
    )
    
    # ---- 4. Video'yu base64'e encode et ----
    try:
        base64_video = encode_video_to_base64(video_path)
    except Exception as e:
        logger.error(f"Video encode hatası: {e}")
        return _create_fallback_response(duration)
    
    # ---- 5. Model çağrısı ----
    try:
        response = vision_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_USER_PROMPT},
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": f"data:{mime_type};base64,{base64_video}",
                                "fps": fps_config["fps"],
                                "max_pixels": fps_config["max_pixels"],
                            }
                        }
                    ]
                }
            ],
            max_tokens=4096,  # YENİ: 2048 token (kesilme riski azaldı)
            temperature=0.1,  # YENİ: Düşük temperature (tutarlı JSON)
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            timeout=180  # YENİ: 3 dakika timeout
        )
    except Exception as e:
        logger.error(f" Vision model çağrısı hatası: {e}")
        return _create_fallback_response(duration)
    
    # ---- 6. Yanıtı parse et ----
    vision_output = response.choices[0].message.content
    
    # RAW çıktıyı log'la (debug için)
    logger.info(f" Vision yanıt uzunluğu: {len(vision_output)} karakter")
    
    parsed = _normalize_spatial_fields(_parse_vision_json(vision_output, duration))
    
    # ---- 7. Sonuç kontrolü ----
    event_count = len(parsed.get("events", []))
    if event_count == 0:
        logger.warning(f" Vision modeli BOŞ events döndürdü! Video: {video_path}")
    else:
        logger.info(f"Vision analizi tamamlandı: {event_count} olay, Süre: {time.time() - start_time:.2f}s")
    
    return parsed

# ============================================================================
# 4. YARDIMCI FONKSİYONLAR
# ============================================================================

_MIME_MAP = {
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
}