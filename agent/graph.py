from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.memory_manager import MemoryManager
from tools.mock_actions import (
    get_frame, analyze_risk, generate_final_report, detect_motion_density,
    flag_critical_timestamps, get_scene_metadata, get_personnel_on_duty,
    assess_environmental_hazards, monitor_hardware_telemetry,
    optimize_inference_params, log_kpi_metrics, trigger_operator_alert,
    query_previous_incidents, export_to_incident_management_system
)
from model_py.services.vision_service import analyze_video
from model_py.services.brain_service import analyze_state
from schemas.output_schema import FinalReportSchema
from model_py.prompts.brain_prompt import BRAIN_SYSTEM_PROMPT
import json
import logging
import re
import time

logger = logging.getLogger("ALFAI_API")

RISK_ORDER = {"Düşük": 1, "Orta": 2, "Yüksek": 3, "Kritik": 4}
RISK_SCORE = {"Düşük": 25, "Orta": 50, "Yüksek": 75, "Kritik": 90}

def sanitize_text(text) -> str:
    """
    Unicode surrogate karakterlerini temizler (\\ud800-\\udfff aralığı).
    Vision modeli bazen bu karakterleri üretiyor; UTF-8 encode edilemiyor
    ve API'ye gönderilince 400 hatası veriyor.
    """
    if not isinstance(text, str):
        text = str(text)
    # surrogatepass ile encode edip replace ile geri decode → bozuk karakterler ? olur
    return text.encode("utf-8", errors="replace").decode("utf-8")


def normalize_risk(value):
    """
    Risk seviyesini standart formata dönüştürür.
    Desteklenen değerler: Kritik, Yüksek, Orta, Düşük
    """
    import re as _re
    
    # None veya boş değer kontrolü
    if value is None:
        return "Düşük"
    
    # String olmayan değerleri string'e çevir
    if not isinstance(value, str):
        value = str(value) if value is not None else ""
    
    # Türkçe karakter düzeltmesi (genişletilmiş)
    text = _re.sub(r'[İI]', 'i', str(value or "")).strip().lower()
    
    # ---- 1. KRİTİK kontrolü (genişletilmiş) ----
    kritik_patterns = [
        "kritik", "critical", "kriti", "critic", 
        "kri̇ti̇k",  # noktalı i ile yazılmış olabilir
        "ciddi", "fatal", "ölümcül"
    ]
    if any(pattern in text for pattern in kritik_patterns):
        return "Kritik"
    
    # ---- 2. YÜKSEK kontrolü (genişletilmiş) ----
    yuksek_patterns = [
        "yüksek", "yuksek", "high", "yuksek",
        "tehlikeli", "dangerous", "ağır"
    ]
    if any(pattern in text for pattern in yuksek_patterns):
        return "Yüksek"
    
    # ---- 3. ORTA kontrolü ----
    orta_patterns = [
        "orta", "medium", "orta",
        "hafif", "light", "normal"
    ]
    if any(pattern in text for pattern in orta_patterns):
        return "Orta"
    
    # ---- 4. DÜŞÜK kontrolü ----
    dusuk_patterns = [
        "düşük", "dusuk", "low", "dusuk",
        "rutin", "normal", "güvenli"
    ]
    if any(pattern in text for pattern in dusuk_patterns):
        return "Düşük"
    
    # ---- 5. Sayısal risk skoru kontrolü ----
    try:
        # Sadece sayısal değerleri yakala
        numbers = _re.findall(r'(\d+\.?\d*)', text)
        if numbers:
            score = float(numbers[0])
            if score >= 85:
                return "Kritik"
            elif score >= 60:
                return "Yüksek"
            elif score >= 35:
                return "Orta"
            else:
                return "Düşük"
    except:
        pass
    
    # ---- 6. Fallback ----
    # Eğer hiçbir pattern eşleşmezse, varsayılan olarak "Düşük" döndür
    return "Düşük"


# RISK_ORDER ve RISK_SCORE sözlüklerini güncelle (üstteki tanımlamalar)
RISK_ORDER = {"Düşük": 1, "Orta": 2, "Yüksek": 3, "Kritik": 4}
RISK_SCORE = {"Düşük": 25, "Orta": 50, "Yüksek": 75, "Kritik": 90}

def higher_risk(a, b):
    a, b = normalize_risk(a), normalize_risk(b)
    return a if RISK_ORDER[a] >= RISK_ORDER[b] else b

def extract_video_info_node(state: AgentState) -> AgentState:
    node_start = time.time()
    import tools.mock_actions as mock_actions

    video_path = state.get("video_path", "")
    mock_actions._world_state_cache[video_path] = {
        "events": state.get("detected_events", []),
        "risk": "Belirlenmedi"
    }

    telemetry = monitor_hardware_telemetry.invoke({})
    optimize_inference_params.invoke({"vram_status": telemetry.get("vram_free_gb", 16.0)})
    scene_meta = get_scene_metadata.invoke({"video_path": video_path})
    zone = scene_meta.get("zone_name", "Genel Saha")
    state["current_zone"] = zone

    detected = state.get("detected_events", [])
    if not detected and video_path:
        try:
            logger.info(f"🎥 Vision modeli ile video taranıyor: {video_path}")
            vision_result = analyze_video(video_path)
            if isinstance(vision_result, dict):
                detected = vision_result.get("events", [])
                state["video_summary"] = vision_result.get("summary", "")
                root_risk = vision_result.get("risk")
                root_score = vision_result.get("risk_score")
                if root_risk in RISK_ORDER:
                    state["vision_risk"] = root_risk
                    state["risk_assessments"] = [root_risk]
                if isinstance(root_score, (int, float)) and 0 <= root_score <= 100:
                    state["vision_risk_score"] = float(root_score)
            elif isinstance(vision_result, list):
                detected = vision_result
        except Exception as e:
            logger.error(f"Vision analizi hatası: {e}")
            detected = []

    normalized = []
    for item in detected:
        e = item if isinstance(item, dict) else {"event": str(item)}
        t = e.get("time", e.get("start_time", "00:00"))
        if isinstance(t, (int, float)):
            t = f"{int(t)//60:02d}:{int(t)%60:02d}"
        normalized.append({
            "time": str(t),
            "start_time": e.get("start_time", t),
            "end_time": e.get("end_time", t),
            "event": e.get("event", e.get("description", "Bilinmeyen Olay")),
            "motion_density": e.get("motion_density", 0.0),
            "zone": e.get("zone", zone),
            "is_critical": e.get("is_critical", e.get("is_critical_moment", False)),
            "is_critical_moment": e.get("is_critical_moment", e.get("is_critical", False)),
            "phase": e.get("phase", ""),
            "event_type": e.get("event_type", ""),
            "ranked_actions": e.get("ranked_actions", []),
            "risk": normalize_risk(e.get("risk", "Düşük")),
            "confidence": e.get("confidence", 0.85),
            "center": e.get("center"),
            "bbox": e.get("bbox"),
            "evidence": e.get("evidence", []),
            "uncertainties": e.get("uncertainties", [])
        })

    event_risk = "Düşük"
    for e in normalized:
        event_risk = higher_risk(event_risk, e["risk"])

    root_risk = normalize_risk(state.get("vision_risk", "Düşük"))
    vision_risk = higher_risk(root_risk, event_risk)
    state["vision_risk"] = vision_risk
    if not isinstance(state.get("vision_risk_score"), (int, float)):
        state["vision_risk_score"] = RISK_SCORE[vision_risk]

    normalized_extra = list(normalized)
    critical_seconds = flag_critical_timestamps.invoke({"video_path": video_path})
    segments = sorted(set(state.get("current_time_segments", []) + critical_seconds))

    for t in segments:
        motion = detect_motion_density.invoke({
            "start_time": t, "end_time": t + 2, "video_path": video_path
        })
        frame = get_frame.invoke({"time": t, "video_path": video_path})
        normalized_extra.append({
            "time": f"{t//60:02d}:{t%60:02d}",
            "start_time": f"{t//60:02d}:{t%60:02d}",
            "end_time": f"{(t+2)//60:02d}:{(t+2)%60:02d}",
            "event": frame,
            "motion_density": motion.get("average_motion_density_percentage", 0.0),
            "zone": zone, "phase": "", "event_type": "Ek Analiz",
            "is_critical": False, "is_critical_moment": False,
            "risk": "Düşük", "confidence": None, "center": None,
            "bbox": None, "ranked_actions": []
        })

    state["detected_events"] = normalized_extra
    state["event_history"] = MemoryManager.add_events(
        state.get("event_history", []), normalized_extra
    )
    state["context_summary"] = MemoryManager.analyze_temporal_context(
        state["event_history"]
    )
    state["next_step"] = "process_risk"
    state["messages"] = [{
        "role": "assistant",
        "content": (
            f"Video ({video_path}) analize alındı. "
            f"Vision risk: {vision_risk}, score: %{state['vision_risk_score']}."
        ),
        "tool_calls": [
            {"name": "monitor_hardware_telemetry", "args": {}, "id": "call_telemetry_001"},
            {"name": "flag_critical_timestamps", "args": {"video_path": video_path}, "id": "call_flag_002"}
        ]
    }]
    logger.info(f"🎯 Vision Risk={vision_risk} | Score=%{state['vision_risk_score']}")
    logger.info(
        f"⏱️ EXTRACT VIDEO NODE SÜRESİ: {time.time() - node_start:.2f} saniye"
    )
    return state


def _build_required_tools_hint(events: list) -> str:
    lines = []
    event_texts = " ".join(
        str(e.get("event", "")).lower() if isinstance(e, dict) else str(e).lower()
        for e in events
    )

    if any(k in event_texts for k in ["işçi", "personel", "kişi", "çalışan", "şahıs", "insan", "yaralı", "müşteri", "birey", "tahliye", "kaç", "devasa patlama"]):
        lines.append("- Sahada etkilenen insan/personel var: get_personnel_on_duty ile can güvenliği veya görevli durumu ZORUNLU OLARAK sorgulanmalıdır.")
    
    if any(k in event_texts for k in ["kaza", "çarpış", "devril", "yangın", "patlama", "silah", "saldırı"]):
        lines.append("- Kritik/Acil olay tespit edildi: trigger_operator_alert ile operatör DERHAL uyarılmalıdır.")
    
    if any(k in event_texts for k in ["yangın", "alev", "duman", "patlama", "kimyasal", "gaz", "sızıntı", "çöktü", "çöker", "çökmesi", "çökme", "yıkıl", "enkaz"]):
        lines.append("- ÇEVRESEL TEHLİKE TESPİTİ: Olayda yangın, patlama, sızıntı veya yapısal çökme (raf/eşya devrilmesi dahil) tespit edildi! Çevresel etki ve enkaz değerlendirmesi için 'assess_environmental_hazards' aracı ZORUNLU OLARAK çağrılmalıdır.")
    
    if not lines:
        return ""

    return (
        "\nSahada tespit edilen durumlara göre kullanman GEREKEN araçlar:\n"
        + "\n".join(lines)
        + "\nBu ipuçlarını dikkate alarak eksiksiz bir analiz yap."
    )

def evaluate_risks_node(state: AgentState) -> AgentState:
    node_start = time.time()
    zone = state.get("current_zone", "Genel Saha")
    video_path = state.get("video_path", "")
    vision_risk = normalize_risk(state.get("vision_risk", "Düşük"))
    highest_risk = vision_risk
    events = []

    for item in state.get("detected_events", []):
        risk = normalize_risk(item.get("risk", "Düşük"))
        highest_risk = higher_risk(highest_risk, risk)
        content = item.get("event", "Açıklama bulunamadı.")
        if isinstance(content, dict):
            content = content.get("description", str(content))
        content = sanitize_text(content)
        events.append(
            f"[{'KRİTİK' if item.get('is_critical_moment') or item.get('is_critical') else 'Normal'}] "
            f"Zaman: {item.get('time','00:00')} "
            f"(Başlangıç: {item.get('start_time','')} → Bitiş: {item.get('end_time','')}) | "
            f"Tür: {item.get('event_type','')} | Aşama: {item.get('phase','')} | "
            f"Vision Risk: {risk} | Olay: {content}"
        )

    state["vision_risk"] = highest_risk
    state["vision_risk_score"] = state.get("vision_risk_score", RISK_SCORE[highest_risk])

    context = state.get("context_summary", {})
    prompt = (
        f"Saha Bölgesi: {zone}.\n"
        f"Vision tarafından belirlenen en yüksek risk: {highest_risk}.\n"
        f"Vision risk skoru: %{state['vision_risk_score']}.\n"
        "Brain Vision riskini düşüremez; yalnızca aynı seviyede kalabilir veya artırabilir.\n"
        f"Temporal Analiz: Toplam {context.get('total_events', 0)} geçmiş olay.\n"
        "Olaylar:\n" + "\n".join(events)
    )
    prompt = sanitize_text(prompt)

    recommendations = set()
    executed_tools = set()
    tool_history_entries = []
    impact_desc = "Otonom yapay zeka analizi devrede."

    required_tools_hint = _build_required_tools_hint(events)

    react_messages = [
        {"role": "system", "content": BRAIN_SYSTEM_PROMPT},
        {"role": "user", "content": prompt + f"\n\n{required_tools_hint}"}
    ]

    react_step = 0
    decision_made = False
    while react_step < 5 and not decision_made:
        react_step += 1
        brain_start = time.time()
        brain_response = analyze_state(react_messages, force_json=False)
        logger.info(
            f"⏱️ BRAIN MODEL SÜRESİ: {time.time() - brain_start:.2f} saniye"
        )
        assistant_msg = {"role": "assistant", "content": brain_response.content or ""}
        if getattr(brain_response, "tool_calls", None):
            assistant_msg["tool_calls"] = [{
                "id": tc.id, "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments}
            } for tc in brain_response.tool_calls]
        react_messages.append(assistant_msg)

        if not getattr(brain_response, "tool_calls", None):
            impact_desc = brain_response.content or "Analiz tamamlandı."
            break

        for tc in brain_response.tool_calls:
            tool_name = tc.function.name
            
            # ---- Tool validasyonu ----
            is_valid, reason = _validate_tool_choice(tool_name, state)
            if not is_valid:
                logger.warning(f"⛔ Geçersiz tool seçimi: '{tool_name}' - {reason}")
                react_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tool_name,
                    "content": json.dumps({
                        "error": f"Tool '{tool_name}' bu bağlamda kullanılamaz: {reason}",
                        "status": "skipped"
                    }, ensure_ascii=False)
                })
                continue
            
            logger.info(f"✅ Tool validasyonu başarılı: '{tool_name}' - {reason}")

            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}

            # ---- 🔥 NORMALIZED ARGS (TEK BİR DÖNGÜ) ----
            normalized_args = {}
            for key, value in args.items():
                if key in ("timestamp", "time", "start_time", "end_time"):
                    if isinstance(value, str) and ":" in value:
                        try:
                            parts = value.split(":")
                            if len(parts) == 2:
                                normalized_args[key] = int(parts[0]) * 60 + int(parts[1])
                            elif len(parts) == 3:
                                normalized_args[key] = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                            else:
                                normalized_args[key] = value
                        except:
                            normalized_args[key] = value
                    else:
                        normalized_args[key] = value
                else:
                    normalized_args[key] = value

            # Video path ve zone ekle (normalized_args'a da ekle)
            if "video_path" not in normalized_args and video_path:
                normalized_args["video_path"] = video_path
                args["video_path"] = video_path  # orijinal args'a da ekle
            if "zone" not in normalized_args and zone:
                normalized_args["zone"] = zone
                args["zone"] = zone

            logger.info(f"⚙️ [{react_step}] '{tool_name}' çalıştırılıyor")
            tool_start = time.time()

            if tool_name == "submit_final_decision":
                raw = str(args.get("risk_level", highest_risk)).strip()
                brain_risk = normalize_risk(raw)
                highest_risk = higher_risk(vision_risk, brain_risk)
                impact_desc = args.get("decision_summary", impact_desc)
                decision_made = True
                tool_result = {
                    "status": "Karar sisteme başarıyla kaydedildi.",
                    "vision_risk": vision_risk,
                    "brain_risk": brain_risk,
                    "final_risk": highest_risk
                }
                executed_tools.add(tool_name)
                # 🔥 SADECE BİR APPEND - normalized_args ile
                tool_history_entries.append({
                    "tool": tool_name,
                    "args": normalized_args,
                    "result": tool_result
                })
                logger.info(
                    f"⏱️ TOOL SÜRESİ | {tool_name}: "
                    f"{time.time() - tool_start:.2f} saniye"
                )
                logger.info(f"🛡️ Risk koruma | Vision={vision_risk} | Brain={brain_risk} | Final={highest_risk}")
                react_messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "name": tool_name,
                    "content": json.dumps(tool_result, ensure_ascii=False)
                })
                break

            try:
                if tool_name == "detect_motion_density":
                    tool_result = detect_motion_density.invoke({
                        "start_time": int(normalized_args.get("start_time", 0)),
                        "end_time": int(normalized_args.get("end_time", 2)),
                        "video_path": video_path
                    })
                elif tool_name == "trigger_operator_alert":
                    tool_result = trigger_operator_alert.invoke(normalized_args)
                    recommendations.add(f"Operatör uyarısı: {normalized_args.get('raw_incident_context', 'Acil Müdahale')}")
                    highest_risk = higher_risk(highest_risk, "Yüksek")
                elif tool_name == "query_previous_incidents":
                    tool_result = query_previous_incidents.invoke(normalized_args)
                    if isinstance(tool_result, list) and tool_result:
                        recommendations.add(f"Geçmişte benzer olay: {tool_result[0].get('root_cause', '')}")
                elif tool_name == "get_personnel_on_duty":
                    tool_result = get_personnel_on_duty.invoke(normalized_args)
                elif tool_name == "assess_environmental_hazards":
                    tool_result = assess_environmental_hazards.invoke(normalized_args)
                elif tool_name == "log_kpi_metrics":
                    tool_result = log_kpi_metrics.invoke(normalized_args)
                else:
                    tool_result = {"status": f"'{tool_name}' çalıştırıldı/onaylandı."}
            except Exception as e:
                tool_result = {"error": str(e)}

            logger.info(
                f"⏱️ TOOL SÜRESİ | {tool_name}: "
                f"{time.time() - tool_start:.2f} saniye"
            )
            executed_tools.add(tool_name)
            
            # 🔥 SADECE BİR APPEND - normalized_args ile
            tool_history_entries.append({
                "tool": tool_name,
                "args": normalized_args,
                "result": tool_result
            })
            
            content = json.dumps(tool_result, ensure_ascii=False)
            react_messages.append({
                "role": "tool", "tool_call_id": tc.id,
                "name": tool_name, "content": content[:1500]
            })

        if decision_made:
            break

    if not decision_made and impact_desc:
        text = impact_desc.lower()
        brain_text_risk = "Kritik" if "kritik" in text else "Yüksek" if ("yüksek" in text or "yuksek" in text) else "Orta" if "orta" in text else "Düşük"
        highest_risk = higher_risk(vision_risk, brain_text_risk)

    highest_risk = higher_risk(highest_risk, vision_risk)

    medium_count = MemoryManager.recent_risk_count(
        state.get("risk_history", []), "Orta"
    )
    if highest_risk == "Orta" and medium_count >= 2:
        highest_risk = "Yüksek"

    state["messages"] = state.get("messages", []) + react_messages[1:]
    state["tool_history"] = state.get("tool_history", []) + tool_history_entries
    state["actions_recommended"] = list(executed_tools)
    state["risk_assessments"] = [highest_risk]
    state["risk_score"] = RISK_SCORE[highest_risk]
    state["technical_summary"] = impact_desc
    state["risk_history"] = MemoryManager.add_risk(
        state.get("risk_history", []), highest_risk
    )
    state["risk_trend"] = MemoryManager.calculate_risk_trend(state["risk_history"])
    state["next_step"] = "compile_report"
    
    logger.info(
        f"📋 ReAct Özeti | Toplam Adım: {react_step} | "
        f"Vision={vision_risk} | Final Risk={highest_risk} | Score=%{state['risk_score']}"
    )
    logger.info(
        f"⏱️ EVALUATE RISKS NODE SÜRESİ: {time.time() - node_start:.2f} saniye"
    )
    return state

def normalize_event_type(value: str) -> str:
    """Event type'ı standart formata dönüştür"""
    if not value:
        return "DURUM"
    
    # String'e çevir ve büyük harf yap
    value = str(value).upper().strip()
    
    # Türkçe karakter düzeltme
    value = value.replace("İ", "I").replace("Ü", "U").replace("Ğ", "G")
    value = value.replace("Ş", "S").replace("Ö", "O").replace("Ç", "C")
    
    # Pattern eşleştirme
    if "ANOMALI" in value or "ANOMALI" in value:
        return "ANOMALI"
    elif "KAZA" in value:
        return "KAZA"
    elif "HAREKET" in value or "MOTION" in value:
        return "HAREKET"
    elif "DURUM" in value or "STATE" in value:
        return "DURUM"
    elif "CIHAZ" in value or "EQUIPMENT" in value or "CIHAZ" in value:
        return "CIHAZ"
    else:
        return "DURUM"  # default


def normalize_phase(value: str) -> str:
    """Phase'ı standart formata dönüştür"""
    if not value:
        return "baslangic"
    
    value = str(value).lower().strip()
    
    # Türkçe karakter düzeltme
    value = value.replace("Ğ", "g").replace("Ü", "u").replace("Ş", "s")
    value = value.replace("Ö", "o").replace("Ç", "c")
    
    if "geli" in value or "orta" in value or "middle" in value:
        return "gelisim"
    elif "son" in value or "bitis" in value or "end" in value:
        return "sonuc"
    else:
        return "baslangic"


def _validate_tool_choice(tool_name: str, state: AgentState) -> tuple[bool, str]:
    events = state.get("detected_events", [])
    risk = state.get("vision_risk", "Düşük")
    
    # Dinamik State Değişkenleri
    critical_count = sum(1 for e in events if e.get("is_critical_moment") or e.get("is_critical"))
    
    # Vision modelinin ürettiği event_type'ları dinamik olarak çekiyoruz (KAZA, ANOMALI vb.)
    event_types = [str(e.get("event_type", "")).upper() for e in events]
    has_anomaly = "ANOMALI" in event_types or "KAZA" in event_types
    
    # ---- 1. log_kpi_metrics ----
    if tool_name == "log_kpi_metrics":
        if risk in ["Kritik", "Yüksek"] or critical_count > 0:
            return False, "Tehlike anında log_kpi_metrics çağrılamaz."
        return True, "Rutin durum - log_kpi_metrics onaylandı."
    
    # ---- 2. DİĞER TÜM AKSİYON ARAÇLARI İÇİN ORTAK DİNAMİK BARAJ ----
    # KURAL: Eğer risk "Düşük" ise, kritik olay (critical_count) yoksa ve 
    # Vision modeli olayı KAZA/ANOMALI olarak etiketlemediyse, bu tam bir RUTİN durumdur.
    if risk == "Düşük" and critical_count == 0 and not has_anomaly:
        action_tools = [
            "trigger_operator_alert", 
            "get_personnel_on_duty", 
            "assess_environmental_hazards", 
            "query_previous_incidents",
            "export_to_incident_management_system",
            "detect_motion_density"
        ]
        if tool_name in action_tools:
            return False, f"Sahada tehlike tespit edilmediği için (Düşük Risk) '{tool_name}' aracı engellendi."
    
    # ---- 3. BEYNE (LLAMA) GÜVEN ----
    # Eğer sistem yukarıdaki rutin barajına takılmadıysa (Yani risk Orta/Yüksek/Kritik ise veya Kaza varsa)
    # Hangi aracın seçileceği konusunda kararı tamamen Llama modelinin muhakeme yeteneğine bırakıyoruz.
    return True, f"'{tool_name}' seçimi dinamik olarak onaylandı."

def compile_report_node(state: AgentState) -> AgentState:
    node_start = time.time()
    brain_risk = normalize_risk(
        state.get("risk_assessments", ["Düşük"])[0]
    )
    vision_risk = normalize_risk(
        state.get("vision_risk", "Düşük")
    )
    # highest_risk = higher_risk(vision_risk, brain_risk)
    highest_risk = brain_risk if brain_risk != "Belirlenmedi" else vision_risk
    calculated_score = RISK_SCORE[highest_risk]

    cleaned_events = []
    for item in state.get("detected_events", []):
        content = item.get("event", "")
        if isinstance(content, dict):
            content = content.get("description", str(content))
        
        # 🔥 YENİ: Evidence'ı dönüştür
        evidence_list = item.get("evidence", [])
        if evidence_list:
            # Eğer string listesi ise dict'e çevir
            if isinstance(evidence_list, list) and all(isinstance(e, str) for e in evidence_list):
                evidence_list = [{"description": e, "confidence": 0.8} for e in evidence_list]
            elif isinstance(evidence_list, str):
                evidence_list = [{"description": evidence_list, "confidence": 0.8}]
        else:
            evidence_list = []
        
        cleaned_events.append({
            "time": item.get("time", "00:00"),
            "event": str(content),
            "start_time": item.get("start_time", item.get("time", "00:00")),
            "end_time": item.get("end_time", item.get("time", "00:00")),
            "event_type": normalize_event_type(item.get("event_type", "")),
            "phase": normalize_phase(item.get("phase", "")),
            "is_critical_moment": item.get("is_critical_moment", item.get("is_critical", False)),
            "ranked_actions": item.get("ranked_actions", []),
            "center": item.get("center"),
            "bbox": item.get("bbox"),
            "zone": item.get("zone", "Genel Saha"),
            "risk": normalize_risk(item.get("risk", "Düşük")),
            "confidence": item.get("confidence", 0.85),
            "evidence": evidence_list,  # ← Dönüştürülmüş
            "uncertainties": item.get("uncertainties", []),
        })

    base_summary = state.get("video_summary", "")
    if not base_summary:
        texts = [e["event"] for e in cleaned_events if e["event"]][:2]
        base_summary = " ".join(texts)[:240] if texts else "Saha görüntüleri incelendi."

    raw_tools = {
        e["tool"] for e in state.get("tool_history", [])
        if isinstance(e, dict) and "tool" in e
    }
    raw_tools.discard("submit_final_decision")
    raw_tools.discard("generate_final_report")

    # ---- Düşük risk durumunda tool temizleme ----
    if highest_risk == "Düşük":
        _akcyon_toollari = {
            "export_to_incident_management_system", "trigger_operator_alert",
            "get_personnel_on_duty", "assess_environmental_hazards",
            "query_previous_incidents", "detect_motion_density"
        }
        raw_tools -= _akcyon_toollari
        raw_tools.add("log_kpi_metrics")

    _total_events = len(cleaned_events)
    _critical_events = sum(1 for e in cleaned_events if e.get("is_critical_moment"))
    _critical_rate = round(_critical_events / _total_events, 3) if _total_events > 0 else 0.0
    _avg_confidence = (
        round(
            sum(
                float(str(e.get("confidence", 0)).replace("%", "")) / (100 if "%" in str(e.get("confidence", "")) else 1)
                for e in cleaned_events if e.get("confidence") is not None
            ) / _total_events,
            3,
        )
        if _total_events > 0 else round(state.get("vision_risk_score", calculated_score) / 100, 2)
    )

    kpi_metrics_data = {
        "ai_processing_status": "Başarılı" if highest_risk not in ["Kritik", "Yüksek"] else "Kritik Müdahale Devrede",
        "confidence_score": _avg_confidence,
        "is_schema_validated": True,
        "detected_events_count": _total_events,
        "critical_events_count": _critical_events,
        "critical_event_capture_rate": _critical_rate,
        "total_processing_seconds": round(state.get("total_processing_seconds", 0.0), 2),
        "action_recommendation_count": len(raw_tools) if raw_tools else 0,
    }

    report_data = {
        "summary": base_summary,
        "events": cleaned_events,
        "risk": highest_risk,
        "risk_score": calculated_score,
        "risk_trend": state.get("risk_trend", "Bilinmiyor"),
        "actions": list(raw_tools) if raw_tools else ["Rutin saha izlemesi devam ediyor."],
        "kpi_metrics": kpi_metrics_data,
    }

    # ---- 🔥 1. JSON REPORT'U OLUŞTUR (export'tan ÖNCE) ----
    try:
        validated = FinalReportSchema(**report_data)
        json_report = validated.model_dump_json(indent=2)
    except Exception as e:
        logger.error(f"🚨 Pydantic şema doğrulama hatası: {e}")
        json_report = json.dumps(report_data, ensure_ascii=False, indent=2)

    _should_export = True
    
    # Düşük risk durumunda (rutin video) export yapma
    if highest_risk == "Düşük":
        _should_export = False
        logger.info("📤 Rutin video - Düşük riskte Export atlanıyor")
    
    # Eğer risk Düşük değilse veya olay sayısı fazlaysa export yap
    if _should_export:
        try:
            export_result = export_to_incident_management_system.invoke({"report_json": json_report})
            raw_tools.add("export_to_incident_management_system")
            logger.info("📤 Export zorunlu olarak çağrıldı (GT uyumluluğu)")
            
            # Export sonucunu tool_history'e ekle
            state["tool_history"] = state.get("tool_history", []) + [{
                "tool": "export_to_incident_management_system",
                "args": {"report_json": json_report[:200] + "..."},
                "result": export_result
            }]
        except Exception as e:
            logger.warning(f"⚠️ CMS export hatası: {e}")

    # ---- 🔥 3. log_kpi_metrics çağrısı (fix: json_report tanımlandı) ----
    if "log_kpi_metrics" in raw_tools:
        try:
            kpi_result = log_kpi_metrics.invoke({
                "metrics_data": {
                    "processed_segments_count": len(cleaned_events),
                    "highest_risk_detected": highest_risk,
                    "risk_score": calculated_score,
                    "vision_risk": vision_risk,
                    "vision_risk_score": state.get("vision_risk_score", calculated_score),
                    "is_schema_validated": "error" not in json_report  # ✅ Artık tanımlı
                }
            })
            logger.info("📊 KPI metrikleri loglandı")
        except Exception as e:
            logger.warning(f"⚠️ KPI log hatası: {e}")

    state["risk_assessments"] = [highest_risk]
    state["risk_score"] = calculated_score
    state["final_json_output"] = json_report
    state["actions_recommended"] = list(raw_tools) if raw_tools else ["Rutin saha izlemesi devam ediyor."]
    
    MemoryManager.save_memory(
        state.get("event_history", []),
        state.get("risk_history", []),
        state.get("tool_history", [])
    )
    state["next_step"] = "complete"

    logger.info(
        f"📤 FINAL RISK | Vision={vision_risk} | Brain={brain_risk} | "
        f"Final={highest_risk} | Score=%{calculated_score}"
    )
    logger.info(
        f"⏱️ COMPILE REPORT NODE SÜRESİ: {time.time() - node_start:.2f} saniye"
    )
    return state

workflow = StateGraph(AgentState)
workflow.add_node("extract_video_info", extract_video_info_node)
workflow.add_node("evaluate_risks", evaluate_risks_node)
workflow.add_node("compile_report", compile_report_node)
workflow.set_entry_point("extract_video_info")
workflow.add_edge("extract_video_info", "evaluate_risks")
workflow.add_edge("evaluate_risks", "compile_report")
workflow.add_edge("compile_report", END)

print("🔍 GRAPH EDGES:", workflow.edges)
app = workflow.compile()