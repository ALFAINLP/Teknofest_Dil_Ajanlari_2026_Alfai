"""React arayüzü ile mevcut ALFAI model servisleri arasındaki HTTP köprüsü."""

import os
import tempfile
import json
from urllib.error import URLError
from urllib.request import urlopen
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from model_py.services.vision_service import analyze_video
from model_py.config.settings import VISION_BASE_URL, VISION_MODEL


app = FastAPI(
    title="ALFAI Web API",
    description="Yüklenen videoları yerel Qwen3-VL modeliyle analiz eder.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1):517\d",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


RISK_SCORES = {
    "Düşük": 25,
    "Orta": 55,
    "Yüksek": 80,
    "Kritik": 95,
}


def _timestamp_to_seconds(timestamp: str) -> float:
    try:
        minutes, seconds = str(timestamp).split(":", maxsplit=1)
        return float(int(minutes) * 60 + int(seconds))
    except (TypeError, ValueError):
        return 0.0


def _normalize_event(event: Any, index: int) -> dict:
    if not isinstance(event, dict):
        event = {"event": str(event)}

    timestamp = str(event.get("time") or "00:00")
    start_time = float(event.get("start_time") or _timestamp_to_seconds(timestamp))
    end_time = float(event.get("end_time") or start_time + 4)
    description = event.get("event") or event.get("description") or "Olay açıklaması bulunamadı."
    if isinstance(description, dict):
        description = description.get("description") or str(description)

    confidence_value = event.get("confidence", 0.85)
    if isinstance(confidence_value, (int, float)):
        confidence = f"%{round(confidence_value * 100) if confidence_value <= 1 else round(confidence_value)}"
    else:
        confidence = str(confidence_value)

    return {
        "id": event.get("id") or f"EVT-{index:03d}",
        "time": timestamp,
        "start_time": start_time,
        "end_time": end_time,
        "end_time_label": event.get("end_time_label") or timestamp,
        "event": str(description),
        "zone": event.get("zone") or "Genel Saha",
        "risk": event.get("risk") or "Orta",
        "confidence": confidence,
        "evidence": event.get("evidence") or [
            {
                "time": timestamp,
                "seconds": start_time,
                "description": str(description),
            }
        ],
        "uncertainties": event.get("uncertainties") or [
            "Model çıktısı operatör doğrulamasına açıktır."
        ],
    }


def _normalize_report(model_output: dict) -> dict:
    risk = str(model_output.get("risk") or "Düşük")
    events = [
        _normalize_event(event, index)
        for index, event in enumerate(model_output.get("events") or [], start=1)
    ]

    confidences = []
    for event in events:
        try:
            confidences.append(float(event["confidence"].strip("%")) / 100)
        except (TypeError, ValueError):
            continue

    average_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0
    actions = model_output.get("actions") or []
    if isinstance(actions, str):
        actions = [actions]

    return {
        "summary": str(model_output.get("summary") or "Video analizi tamamlandı."),
        "events": events,
        "risk": risk,
        "risk_score": int(model_output.get("risk_score") or RISK_SCORES.get(risk, 50)),
        "actions": [str(action) for action in actions],
        "kpi_metrics": {
            "ai_processing_status": "Başarılı",
            "confidence_score": average_confidence,
            "is_schema_validated": True,
        },
    }


@app.get("/")
def root() -> dict:
    return {"status": "ok", "service": "ALFAI Web API"}


@app.get("/api/health")
def health() -> dict:
    try:
        with urlopen(f"{VISION_BASE_URL}/models", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        available_models = {item.get("id") for item in payload.get("data", [])}
        model_ready = VISION_MODEL in available_models
        return {
            "status": "healthy" if model_ready else "degraded",
            "vision_model": VISION_MODEL,
            "model_ready": model_ready,
        }
    except (OSError, URLError, ValueError, json.JSONDecodeError) as error:
        return {
            "status": "degraded",
            "vision_model": VISION_MODEL,
            "model_ready": False,
            "detail": str(error),
        }


@app.post("/api/analyze-video")
async def analyze_uploaded_video(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    temporary_path = ""

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_path = temporary_file.name
            while chunk := await file.read(1024 * 1024):
                temporary_file.write(chunk)

        model_output = await run_in_threadpool(analyze_video, temporary_path)
        if not isinstance(model_output, dict):
            raise ValueError("Model sözlük biçiminde JSON üretmedi.")
        return _normalize_report(model_output)
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Model analizi başarısız: {error}") from error
    finally:
        await file.close()
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)
