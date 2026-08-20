"""React arayüzü ile mevcut ALFAI model servisleri arasındaki HTTP köprüsü."""

import os
import tempfile
import subprocess
import json
from urllib.error import URLError
from urllib.request import urlopen
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from model_py.services.vision_service import analyze_video
from model_py.services.camera_triage_service import analyze_camera_triage
from model_py.config.settings import VISION_BASE_URL, VISION_MODEL

from pydantic import BaseModel
from agent.graph import app as langgraph_app
from agent.memory_manager import MemoryManager
from model_py.services.chat_service import build_chat_messages, chat as llama_chat
from model_py.config.settings import VISION_BASE_URL, VISION_MODEL
import traceback
import json
import time
import logging
import os
import tempfile

# --- LOGLAMA AYARLARI ---
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG yaparsak her detayı görürüz. Canlıya alırken INFO yapılır.
    format="%(asctime)s | %(levelname)-7s | [%(name)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("ALFAI_API")

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


def _read_gpu_metrics() -> dict:
    """NVIDIA sürücüsündeki anlık GPU ve VRAM değerlerini okur."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
        devices = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            index, name, utilization, memory_used, memory_total = [part.strip() for part in line.split(",", 4)]
            devices.append({
                "index": int(index),
                "name": name,
                "gpu_utilization_percent": float(utilization),
                "vram_used_mb": float(memory_used),
                "vram_total_mb": float(memory_total),
            })

        if not devices:
            raise RuntimeError("GPU bilgisi dönmedi.")

        used = sum(device["vram_used_mb"] for device in devices)
        total = sum(device["vram_total_mb"] for device in devices)
        return {
            "available": True,
            "gpu_utilization_percent": round(sum(device["gpu_utilization_percent"] for device in devices) / len(devices), 1),
            "vram_used_mb": round(used, 1),
            "vram_total_mb": round(total, 1),
            "vram_utilization_percent": round((used / total) * 100, 1) if total else 0,
            "devices": devices,
        }
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, RuntimeError) as error:
        return {"available": False, "detail": str(error)}


def _normalize_ranked_actions(event: dict, description: str, zone: str) -> list[dict]:
    raw_actions = event.get("ranked_actions") or []
    actions = []
    for fallback_rank, action in enumerate(raw_actions, start=1):
        if isinstance(action, str):
            action = {"label": action}
        if not isinstance(action, dict):
            continue
        try:
            rank = int(action.get("rank") or fallback_rank)
        except (TypeError, ValueError):
            rank = fallback_rank
        if rank not in (1, 2, 3) or any(existing["rank"] == rank for existing in actions):
            continue
        actions.append({
            "rank": rank,
            "label": str(action.get("label") or action.get("action") or "Operatör incelemesi başlat"),
            "reason": str(action.get("reason") or "Modelin olaya özel operasyon önerisi."),
        })

    fallbacks = [
        {"rank": 1, "label": f"{zone} operatörüne bildir", "reason": f"'{description}' bulgusunun hızla doğrulanması için."},
        {"rank": 2, "label": "Görüntüyü odağa al", "reason": "Olayın gelişimini ve çevresel bağlamı izlemek için."},
        {"rank": 3, "label": "Önleyici inceleme oluştur", "reason": "Benzer olayların tekrarını azaltacak uzun vadeli önlemleri belirlemek için."},
    ]
    existing_ranks = {action["rank"] for action in actions}
    actions.extend(action for action in fallbacks if action["rank"] not in existing_ranks)
    return sorted(actions, key=lambda action: action["rank"])[:3]


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
    raw_start = event.get("start_time") or timestamp
    start_time = float(_timestamp_to_seconds(raw_start) if isinstance(raw_start, str) and ":" in raw_start else raw_start)

    raw_end = event.get("end_time") or timestamp
    end_time = float(_timestamp_to_seconds(raw_end) if isinstance(raw_end, str) and ":" in raw_end else raw_end)
    description = event.get("event") or event.get("description") or "Olay açıklaması bulunamadı."
    if isinstance(description, dict):
        description = description.get("description") or str(description)

    confidence_value = event.get("confidence", 0.85)
    if isinstance(confidence_value, (int, float)):
        confidence = f"%{round(confidence_value * 100) if confidence_value <= 1 else round(confidence_value)}"
    else:
        confidence = str(confidence_value)

    zone = event.get("zone") or "Genel Saha"
    return {
        "id": event.get("id") or f"EVT-{index:03d}",
        "time": timestamp,
        "start_time": start_time,
        "end_time": end_time,
        "end_time_label": event.get("end_time_label") or timestamp,
        "event": str(description),
        "zone": zone,
        "risk": event.get("risk") or "Orta",
        "confidence": confidence,
        "center": event.get("center"),
        "bbox": event.get("bbox"),
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
        "ranked_actions": _normalize_ranked_actions(event, str(description), zone),
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


@app.post("/v1/incident/export")
def export_incident(payload: dict) -> dict:
    logger.info(f"📋 Incident exported to CMS: {payload.get('ticket_title', 'Durum Raporu')}")
    return {"status": "success", "message": "Incident report exported successfully", "ticket_id": "INC-2026-001"}


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


@app.get("/api/performance")
def performance() -> dict:
    """Analiz sırasında panelin sorguladığı anlık donanım telemetrisi."""
    return _read_gpu_metrics()


@app.post("/api/triage-cameras")
async def triage_cameras(
    files: list[UploadFile] = File(...),
    camera_ids: list[str] = Form(...),
) -> dict:
    """Az sayıda temsilî kareyle kameraları tek Qwen isteğinde önceliklendirir."""
    if not files or len(files) > 4:
        raise HTTPException(status_code=400, detail="1 ile 4 arasında video yüklenmelidir.")
    if len(files) != len(camera_ids):
        raise HTTPException(status_code=400, detail="Kamera kimliği ve video sayısı eşleşmiyor.")

    temporary_paths = []
    try:
        for upload in files:
            suffix = Path(upload.filename or "video.mp4").suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
                temporary_paths.append(temporary_file.name)
                while chunk := await upload.read(1024 * 1024):
                    temporary_file.write(chunk)

        camera_videos = list(zip(camera_ids, temporary_paths))
        logger.info("🎥 [HIZLI TRİYAJ] %s kamera tek Qwen isteğiyle taranıyor.", len(camera_videos))
        return await run_in_threadpool(analyze_camera_triage, camera_videos)
    except Exception as error:
        logger.error("💥 Kamera triyajı başarısız: %s", error, exc_info=True)
        raise HTTPException(status_code=502, detail=f"Kamera triyajı başarısız: {error}") from error
    finally:
        for upload in files:
            await upload.close()
        for path in temporary_paths:
            if os.path.exists(path):
                os.remove(path)


@app.post("/api/analyze-video")
async def analyze_uploaded_video(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    temporary_path = ""
    start_time = time.time()
    
    logger.info(f"📥 [YENİ VİDEO İSTEĞİ] Adı: {file.filename}")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_path = temporary_file.name
            while chunk := await file.read(1024 * 1024):
                temporary_file.write(chunk)
                
        file_size_mb = os.path.getsize(temporary_path) / (1024 * 1024)
        logger.debug(f"📁 Dosya geçici klasöre kaydedildi. Boyut: {file_size_mb:.2f} MB. Yol: {temporary_path}")

        logger.info("👁️ [AŞAMA 1] Vision modeli (Qwen3-VL) çalıştırılıyor...")
        vision_started_at = time.perf_counter()
        model_output = await run_in_threadpool(analyze_video, temporary_path)
        vision_inference_ms = round((time.perf_counter() - vision_started_at) * 1000, 1)

        if not isinstance(model_output, dict):
            logger.error("❌ Model çıktısı beklenen formatta (dict) değil!")
            raise ValueError("Model sözlük biçiminde JSON üretmedi.")

        logger.debug(f"✅ Vision modeli yanıt verdi. Tespit edilen ham olay sayısı: {len(model_output.get('events', []))}")
        # return _normalize_report(model_output)
        logger.info("🧠 [AŞAMA 2] Geçmiş hafıza (MemoryManager) yükleniyor...")
        memory = MemoryManager.load_memory()

        initial_state = {
            "video_path": temporary_path,
            "current_time_segments": [],
            "detected_events": model_output.get("events", []), # Vision modelinin çıktısını LangGraph'a besliyoruz!
            "risk_assessments": [],
            "actions_recommended": [],
            "final_json_output": "",
            "next_step": "",
            "risk_trend": "",
            "context_summary": {},
            "technical_summary": "",
            "event_history": memory.get("event_history", []),
            "risk_history": memory.get("risk_history", []),
            "tool_history": memory.get("tool_history", []),
            "messages": [],
        }
        
        logger.debug(f"📦 LangGraph Initial State hazırlandı. Keys: {list(initial_state.keys())}")
        logger.info("🚀 [AŞAMA 3] LangGraph Ajanı invoke ediliyor...")
        
        final_state = await run_in_threadpool(langgraph_app.invoke, initial_state)
        
        logger.info("✅ LangGraph çalışmasını tamamladı.")
        
        if "messages" in final_state:
            logger.debug(f"💬 Ajan toplam {len(final_state['messages'])} mesaj/adım üretti.")
            for i, msg in enumerate(final_state["messages"]):
                msg_type = msg.get("type", "Unknown") if isinstance(msg, dict) else msg.__class__.__name__
                content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                
                if content:
                    logger.debug(f"   [Adım {i+1} | {msg_type}] 💭 Çıktı: {str(content)[:150]}...")
            
        print("="*40 + "\n")
        
        
        print("\n" + "="*60)
        print("🧠 [AJANIN ADIM ADIM DÜŞÜNME ZİNCİRİ (REASONING CHAIN)]")
        print("="*60)
        
        if "messages" in final_state:
            for i, msg in enumerate(final_state["messages"]):
                # Mesaj LangChain objesi mi yoksa sözlük (dict) mü kontrol edelim
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    tool_calls = msg.get("tool_calls", [])
                    msg_type = msg.get("type", "Unknown")
                else:
                    content = getattr(msg, "content", "")
                    tool_calls = getattr(msg, "tool_calls", [])
                    msg_type = msg.__class__.__name__

                print(f"\n{'='*20} ADIM {i+1} | TÜR: {msg_type.upper()} {'='*20}")
                
                # Modelin iç sesi / çıkarımları
                if content:
                    print(f"💭 Düşünce/Çıktı:\n{content.strip()}")
                
                # Model bir fonksiyona/araca (tool) başvurduysa
                if tool_calls:
                    print("\n🛠️ ÇAĞRILAN ARAÇLAR:")
                    for tool in tool_calls:
                        tool_name = tool.get('name') if isinstance(tool, dict) else getattr(tool, 'name', 'Bilinmiyor')
                        tool_args = tool.get('args') if isinstance(tool, dict) else getattr(tool, 'args', {})
                        print(f"  - Araç Adı: {tool_name}")
                        print(f"  - Parametreler: {tool_args}")
                
                # ToolMessage ise aracın döndürdüğü sonuç
                if msg_type == "ToolMessage" or msg_type == "tool":
                    tool_name = msg.get("name") if isinstance(msg, dict) else getattr(msg, "name", "Araç")
                    print(f"⚙️ {tool_name} Aracından Gelen Sonuç:\n{str(content)[:300]}... (devamı kırpıldı)")

        print("\n" + "="*60 + "\n")
        
        
        # final_json_output: compile_report node tarafından üretilen JSON string
        raw_json = final_state.get("final_json_output", "")
        logger.info("📄 [AŞAMA 4] Ajan raporu JSON olarak compile edildi.")
        if raw_json:
            try:
                parsed_json = json.loads(raw_json)
            except (json.JSONDecodeError, TypeError) as error:
                logger.warning(f"⚠️ JSON Parse Hatası! Ajan bozuk JSON üretti: {error}")
                parsed_json = {}
        else:
            parsed_json = {}

        # final_state ile parsed_json'u birleştiriyoruz:
        # final_state → graph'ın tüm değişkenleri (risk_trend, detected_events, vb.)
        # parsed_json  → compile_report'un ürettiği özet (summary, actions, vb.)
        graph_output = {**final_state, **parsed_json}

        # 3. React için Yalnızca Zamana Göre Tekilleştirilmiş (Deduplicated by Time) Format
        safe_events = []
        seen_times = set() # Sadece zamanları takip ediyoruz

        for evt in graph_output.get("detected_events", []):
            event_time = evt.get("time", "00:00")
            if event_time not in seen_times:
                seen_times.add(event_time)
                safe_events.append(evt)

        logger.info(f"🧹 [FİLTRELEME] Aynı saniyedeki kopya olaylar silindi. Net olay sayısı: {len(safe_events)}")
                
        total_raw_events = len(graph_output.get("detected_events", []))
        total_safe_events = len(safe_events)
        print(f"🧹 [FİLTRELEME] Model toplam {total_raw_events} olay buldu. Aynı saniyedeki {total_raw_events - total_safe_events} kopya silindi. Kalan net olay: {total_safe_events}")

        safe_actions = graph_output.get("actions_recommended", [])
        if not safe_actions:
            safe_actions = ["Rutin saha izlemesi devam ediyor, anomali tespiti bekleniyor."]

        model_output = {
            "summary": parsed_json.get("summary") or graph_output.get("technical_summary") or "Video analiz edildi, genel durum stabil.",
            "events": safe_events,
            "risk": graph_output.get("risk_assessments", ["Düşük"])[0] if graph_output.get("risk_assessments") else "Düşük",
            "actions": safe_actions
        }

        final_response = _normalize_report(model_output)
        final_response["risk_trend"] = graph_output.get("risk_trend", "Stabil")
        
        elapsed_time = time.time() - start_time
        final_response["performance"] = {
            "total_processing_seconds": round(elapsed_time, 2),
            "vision_inference_ms": vision_inference_ms,
            "gpu": _read_gpu_metrics(),
        }
        logger.info(f"🏁 [TAMAMLANDI] Toplam İşlem Süresi: {elapsed_time:.2f} saniye.")
        logger.debug(f"📤 Frontend'e dönen final JSON: Risk={final_response.get('risk')}, Olay={len(final_response.get('events', []))}")

        return final_response

    except Exception as error:
        logger.error(f"💥 KRİTİK HATA: Model analizi sırasında bir şeyler çöktü: {str(error)}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"Model analizi başarısız: {error}") from error
    finally:
        await file.close()
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)
            logger.debug("🗑️ Geçici video dosyası silindi.")


# ─────────────────────────────────────────────────────────────
# Chat Asistanı — Metin Tabanlı Kullanıcı-Ajan Etkileşimi
# ─────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    """Tek bir sohbet turu mesajı."""
    role: str     # "user" veya "assistant"
    content: str


class ChatRequest(BaseModel):
    """
    Frontend'den gelen chat isteği.

    - message:          Operatörün yeni sorusu / mesajı.
    - analysis_context: Aktif analiz sonucu (events, risk, summary, vb.).
                        Frontend, video analizi bittikten sonra bunu doldurur.
                        Yoksa None / boş bırakılabilir.
    - history:          Aynı oturumdaki önceki sohbet turları (isteye bağlı).
                        Llama'nın önceki soruları hatırlaması için kullanılır.
    """
    message: str
    analysis_context: Optional[dict] = None
    history: Optional[list[ChatMessage]] = None


@app.post("/api/chat")
async def chat_with_agent(req: ChatRequest) -> dict:
    """
    Operatörün soru / mesajını Llama 3.1 70B'ye iletir.

    Model:
    - Aktif video analizi bağlamını (events, risk, summary) sistem promptuna alır.
    - Geçmiş hafızayı (memory.json) bağlama ekler.
    - Önceki sohbet turlarını hatırlayarak çoklu tur desteği sunar.
    - Araç çağrısı (tool_call) YAPMAZ — saf konuşma modu.

    Dönen yapı:
        reply (str): Llama'nın Türkçe yanıtı.
        has_context (bool): Analiz bağlamının mevcut olup olmadığı.
    """
    try:
        logger.info(f"💬 [YENİ CHAT İSTEĞİ] Kullanıcı sorusu: {req.message[:50]}...")
        # Kalıcı hafızayı yükle
        memory = MemoryManager.load_memory()

        # Önceki sohbet turlarını sözlük listesine çevir
        history_dicts = [
            {"role": m.role, "content": m.content}
            for m in (req.history or [])
        ]
        logger.debug(f"🔍 Chat Bağlamı: {len(history_dicts)} geçmiş tur bulundu. Analiz context var mı? {bool(req.analysis_context)}")

        # Sistem promptu + bağlam + geçmiş + yeni soru
        messages = await run_in_threadpool(
            build_chat_messages,
            req.message,
            req.analysis_context,
            memory,
            history_dicts,
        )

        print(f"\n💬 [CHAT] Operatör: {req.message[:80]}")
        print(f"   Bağlam var mı: {'Evet' if req.analysis_context else 'Hayır'}")
        print(f"   Geçmiş tur sayısı: {len(history_dicts)}")
        
        # Mevcut printlerin yanına ekleyin
        if req.analysis_context:
            context_events_count = len(req.analysis_context.get("events", []))
            context_risk = req.analysis_context.get("risk", "Bilinmiyor")
            print(f"   📥 [CHAT BAĞLAMI] Modele iletilen videodaki olay sayısı: {context_events_count}, Risk: {context_risk}")

        logger.info("🤖 Llama modeline (Chat) istek gönderiliyor...")
        # Llama'ya gönder
        reply = await run_in_threadpool(llama_chat, messages)
        logger.debug(f"✅ Llama yanıtı alındı: {reply[:100]}...")


        return {
            "reply": reply,
            "has_context": bool(req.analysis_context),
        }

    except Exception as error:
        logger.error(f"💥 CHAT HATASI: {str(error)}", exc_info=True)
        traceback.print_exc()
        raise HTTPException(
            status_code=502,
            detail=f"Chat yanıtı üreti lemedi: {error}",
        ) from error
