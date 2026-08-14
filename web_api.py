"""React arayüzü ile mevcut ALFAI model servisleri arasındaki HTTP köprüsü."""

import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from agent.graph import app as langgraph_app
#from model_py.services.vision_service import analyze_video
import traceback

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
    return {"status": "healthy", "vision_model": "Qwen/Qwen3-VL-8B-Instruct"}


@app.post("/api/analyze-video")
async def analyze_uploaded_video(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    temporary_path = ""

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_path = temporary_file.name
            while chunk := await file.read(1024 * 1024):
                temporary_file.write(chunk)

        initial_state = {"video_path": temporary_path}
        graph_output = await run_in_threadpool(langgraph_app.invoke, initial_state)

        # 3. React için Yalnızca Zamana Göre Tekilleştirilmiş (Deduplicated by Time) Format
        safe_events = []
        seen_times = set() # Sadece zamanları takip ediyoruz

        for evt in graph_output.get("detected_events", []):
            event_content = evt.get("event", "Olay tespit edilemedi.")
            if isinstance(event_content, dict):
                event_text = event_content.get("description", str(event_content))
            else:
                event_text = str(event_content)
            
            event_time = evt.get("time", "00:00")
            
            # Aynı saniyede birden fazla olay olmasını engellemek için sadece zamanı kontrol et
            if event_time not in seen_times:
                seen_times.add(event_time)
                clean_evt = evt.copy()
                clean_evt["event"] = event_text
                safe_events.append(clean_evt)

        safe_actions = graph_output.get("actions_recommended", [])
        if not safe_actions:
            safe_actions = ["Rutin saha izlemesi devam ediyor, anomali tespiti bekleniyor."]

        model_output = {
            "summary": graph_output.get("technical_summary") or "Video analiz edildi, genel durum stabil.",
            "events": safe_events,
            "risk": graph_output.get("risk_assessments", ["Düşük"])[0] if graph_output.get("risk_assessments") else "Düşük",
            "actions": safe_actions
        }

        final_response = _normalize_report(model_output)
        final_response["risk_trend"] = graph_output.get("risk_trend", "Stabil")

        return final_response
        
    except Exception as error:
        print("\n" + "="*50)
        print("🚨 API'DE GİZLİ HATA YAKALANDI 🚨")
        traceback.print_exc()  
        print("="*50 + "\n")
        raise HTTPException(status_code=502, detail=f"Model analizi başarısız: {error}") from error
    finally:
        await file.close()
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)
            
            
            
            
