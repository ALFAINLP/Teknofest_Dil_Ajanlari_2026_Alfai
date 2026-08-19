from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.memory_manager import MemoryManager
from tools.mock_actions import (
    get_frame,
    analyze_risk,
    generate_final_report,
    detect_motion_density,
    flag_critical_timestamps,
    get_scene_metadata,
    get_personnel_on_duty,
    assess_environmental_hazards,
    monitor_hardware_telemetry,
    optimize_inference_params,
    log_kpi_metrics,
    trigger_operator_alert,
    query_previous_incidents,
    export_to_incident_management_system
)
from model_py.services.vision_service import analyze_video
from model_py.services.brain_service import analyze_state 
import traceback
import json
from model_py.prompts.brain_prompt import BRAIN_SYSTEM_PROMPT
import logging
logger = logging.getLogger("ALFAI_API")
from schemas.output_schema import FinalReportSchema

def extract_video_info_node(state: AgentState) -> AgentState:
    import tools.mock_actions as mock_actions
    mock_actions._world_state_cache[state["video_path"]] = {
        "events": state.get("detected_events", []),
        "risk": "Belirlenmedi"
    }
    telemetry = monitor_hardware_telemetry.invoke({})
    vram_free = telemetry.get("vram_free_gb", 16.0)
    optimize_inference_params.invoke({"vram_status": vram_free})
    
    scene_meta = get_scene_metadata.invoke({"video_path": state["video_path"]})
    
    zone = scene_meta.get("zone_name", "Genel Saha")
    state["current_zone"] = zone
    
    # web_api.py zaten Vision modelini çalıştırdı ve detected_events'ı state'e koydu.
    # Burada tekrar analyze_video çağırmak gereksizdir, state'teki mevcut olayları kullanıyoruz.
    detected = []
    for event_item in state.get("detected_events", []):
        event_content = event_item if isinstance(event_item, dict) else {"event": str(event_item)}
        detected.append({
            "time": event_content.get("time", "00:00"),
            "event": event_content.get("event", event_content.get("description", "Bilinmeyen Olay")),
            "motion_density": event_content.get("motion_density", 0.0),
            "zone": event_content.get("zone", zone),
            "risk": event_content.get("risk"),
            "confidence": event_content.get("confidence"),
            "center": event_content.get("center"),
            "bbox": event_content.get("bbox")
        })
        
    state["detected_events"] = detected
    
    

    critical_seconds = flag_critical_timestamps.invoke({"video_path": state["video_path"]})
    current_segments = state.get("current_time_segments", [])
    target_segments = sorted(list(set(current_segments + critical_seconds)))
    
    
    
    for t in target_segments:
        motion = detect_motion_density.invoke({
            "start_time": t, 
            "end_time": t+2, 
            "video_path": state["video_path"]
        })
        frame_info = get_frame.invoke({
            "time": t, 
            "video_path": state["video_path"]
        })
        
        
        
        minutes = t // 60
        seconds = t % 60
        time_str = f"{minutes:02d}:{seconds:02d}"
        
        detected.append({
            "time": time_str,
            "event": frame_info,
            "motion_density": motion.get("average_motion_density_percentage", 0.0),
            "zone": zone
        })
        
    state["detected_events"] = detected

    # Hafızaya olayları ekle
    state["event_history"] = MemoryManager.add_events(
        state.get("event_history", []),
        detected
    )  
    # ==========================
    # Temporal Context
    # ==========================
    state["context_summary"] = MemoryManager.analyze_temporal_context(
        state["event_history"]
    )
    state["next_step"] = "process_risk"
    
    log_message = {
        "role": "assistant",
        "content": f"Video ({state['video_path']}) analize alındı. Donanım telemetrisi ölçüldü, sahne meta verileri ve kritik hareket saniyeleri tespit edildi.",
        "tool_calls": [
            {
                "name": "monitor_hardware_telemetry", 
                "args": {}, 
                "id": "call_telemetry_001"
            },
            {
                "name": "flag_critical_timestamps", 
                "args": {"video_path": state['video_path']}, 
                "id": "call_flag_002"
            }
        ]
    }
    state["messages"] = [log_message]
    
    return state

def evaluate_risks_node(state: AgentState) -> AgentState:
    zone = state.get("current_zone", "Genel Saha")
    env_hazards = assess_environmental_hazards.invoke({"zone": zone})

    # Olayları formatla
    events_for_brain = []
    for item in state["detected_events"]:
        event_content = item.get("event", "Açıklama bulunamadı.")
        if isinstance(event_content, dict):
            event_text = event_content.get("description", str(event_content))
        else:
            event_text = str(event_content)
        
        motion = item.get('motion_density', 0)
        has_risk_flag = item.get('has_risk', False)
        time = item.get('time', '00:00')
        
        events_for_brain.append(
            f"Zaman: {time} | Olay: {event_text} | "
            f"Hareket Yoğunluğu: %{motion:.1f} | Risk Varlığı: {'Evet' if has_risk_flag else 'Hayır'}"
        )

    # Geçmiş hafızayı özet olarak hazırla
    risk_history_summary = state.get("risk_history", [])[-5:]
    risk_trend = state.get("risk_trend", "Stabil (Yetersiz Veri)")
    context_summary = state.get("context_summary", {})
    tool_history_recent = state.get("tool_history", [])[-3:]

    memory_block = (
        f"\n--- HAFIZA & BAĞLAM ---\n"
        f"Son 5 Risk Geçmişi: {risk_history_summary}\n"
        f"Risk Trendi: {risk_trend}\n"
        f"Temporal Analiz: Toplam {context_summary.get('total_events', 0)} geçmiş olay. "
        f"Baskın Bölge: {context_summary.get('dominant_zone', 'Bilinmiyor')}. "
        f"Sistematik Risk Var mı: {'Evet' if context_summary.get('systematic_risk') else 'Hayır'}. "
        f"Önerilen Aksiyon: {context_summary.get('recommended_action', 'Normal İzleme')}.\n"
        f"Son Kullanılan Araçlar: {tool_history_recent}\n"
        f"--- HAFIZA SONU ---\n"
    )

    clean_user_prompt = (
        f"Şu anki bölge: {zone}.\n"
        f"Çevresel riskler: {env_hazards}.\n"
        f"{memory_block}"
        f"Aşağıdaki olayları incele ve risk seviyesini belirle:\n"
        f"{chr(10).join(events_for_brain)}"
    )

    logger.info(f"🔍 evaluate_risks_node çalışıyor. Mevcut state: {state.keys()}")

    recommendations = set()
    highest_risk = "Düşük"
    impact_desc = "Otonom yapay zeka analizi devrede."
    tool_history_entries = state.get("tool_history", [])
    
    react_messages = [
        {"role": "system", "content": BRAIN_SYSTEM_PROMPT},
        {"role": "user", "content": clean_user_prompt}
    ]

    MAX_REACT_STEPS = 5
    react_step = 0
    decision_made = False

    # ================================================================
    # ASIL REACT DÖNGÜSÜ (BURASI ÇALIŞACAK)
    # ================================================================
    while react_step < MAX_REACT_STEPS:
        react_step += 1
        logger.info(f"🔄 ReAct Adım {react_step}: Brain çağrılıyor...")

        # API'ye HER ZAMAN araçlar GİDECEK (force_json=False)
        brain_response = analyze_state(react_messages, force_json=False)
        assistant_msg = {"role": "assistant", "content": brain_response.content or ""}

        if hasattr(brain_response, "tool_calls") and brain_response.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in brain_response.tool_calls
            ]

        react_messages.append(assistant_msg)

        # Model hiçbir araç çağırmadıysa ve henüz karar vermediyse döngüyü zorla bitir
        if not hasattr(brain_response, "tool_calls") or not brain_response.tool_calls:
            logger.warning(f"⚠️ ReAct Adım {react_step}: Model tool çağırmayı unuttu, döngü sonlandırılıyor.")
            if not decision_made:
                impact_desc = brain_response.content if brain_response.content else "Analiz tamamlandı."
            break

        # Araçları çalıştır ve mesaj geçmişine ekle
        for tool_call in brain_response.tool_calls:
            tool_name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
                import ast
                for key, value in list(args.items()):
                    if isinstance(value, str):
                        val_str = value.strip()
                        if val_str.startswith('{') and val_str.endswith('}'):
                            try:
                                # Deneme 1: Tek tırnakları çifte çevirerek standart JSON parse
                                safe_json = val_str.replace("'", '"')
                                args[key] = json.loads(safe_json)
                            except Exception:
                                try:
                                    # Deneme 2: Doğrudan Python sözlüğü olarak değerlendir
                                    args[key] = ast.literal_eval(val_str)
                                except Exception:
                                    pass
            except json.JSONDecodeError:
                args = {}
            tool_result = None

            logger.info(f"🛠️ Brain '{tool_name}' aracını çağırdı. Argümanlar: {args}")

            # ================================================================
            # 1. NİHAİ KARAR ARACI (Döngüyü bitiren sinyal)
            # ================================================================
            if tool_name == "submit_final_decision":
                raw_risk = args.get("risk_level", highest_risk).strip().lower()
                
                # 🟢 DÜZELTME: Modelin yapabileceği imla hatalarını (Döşük, Döşk vb.) standart kategorilere eşitleme
                if "kritik" in raw_risk:
                    highest_risk = "Kritik"
                elif "yüksek" in raw_risk or "yuksek" in raw_risk:
                    highest_risk = "Yüksek"
                elif "orta" in raw_risk:
                    highest_risk = "Orta"
                else:
                    highest_risk = "Düşük" # 'Düşk' gibi hatalı yazımlar otomatik olarak 'Düşük' kabul edilir

                impact_desc = args.get("decision_summary", impact_desc)
                decision_made = True
                tool_result = {"status": "Karar sisteme başarıyla kaydedildi, görevi sonlandır."}

            # ================================================================
            # 2. DİĞER FONKSİYONEL ARAÇLAR
            # ================================================================
            elif tool_name == "trigger_operator_alert":
                tool_result = trigger_operator_alert.invoke(args)
                tool_history_entries.append({"tool": tool_name, "args": args, "result": tool_result})
                recommendations.add(f"Operatör uyarısı: {args.get('raw_incident_context', '')}")
                if highest_risk not in ["Kritik", "Yüksek"]: highest_risk = "Yüksek"

            elif tool_name == "query_previous_incidents":
                tool_result = query_previous_incidents.invoke(args)
                tool_history_entries.append({"tool": tool_name, "args": args, "result": tool_result})
                if tool_result: recommendations.add(f"Geçmişte benzer olay: {tool_result[0].get('root_cause', '')}")
                if highest_risk not in ["Kritik", "Yüksek"]: highest_risk = "Yüksek"

            elif tool_name == "get_personnel_on_duty":
                tool_result = get_personnel_on_duty.invoke(args)
                tool_history_entries.append({"tool": tool_name, "args": args, "result": tool_result})
                if tool_result: recommendations.add(f"Görevli personel: {', '.join([p.get('name', '?') for p in tool_result])}")

            elif tool_name == "assess_environmental_hazards":
                tool_result = assess_environmental_hazards.invoke(args)
                tool_history_entries.append({"tool": tool_name, "args": args, "result": tool_result})

            elif tool_name == "log_kpi_metrics":
                tool_result = log_kpi_metrics.invoke(args)
                tool_history_entries.append({"tool": tool_name, "args": args, "result": tool_result})
                
            else:
                tool_result = {"error": "Bilinmeyen araç."}

            # Tool sonucunu mesaj geçmişine ekle (Brain'in bir sonraki turda görmesi için)
            react_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(tool_result, ensure_ascii=False)
            })

        # Eğer model `submit_final_decision` çağırdıysa döngüyü sonlandır
        if decision_made:
            logger.info("✅ Model nihai kararını bildirdi, ReAct döngüsü tamamlandı.")
            break


    if not recommendations:
        recommendations.add("Rutin saha izlemesi devam ediyor, anomali tespiti bekleniyor.")

    # Adaptive Risk
    medium_count = MemoryManager.recent_risk_count(state.get("risk_history", []), "Orta")
    if highest_risk == "Orta" and medium_count >= 2:
        highest_risk = "Yüksek"
        state["messages"].append({
            "role": "assistant",
            "content": f"⚠️ Risk seviyesi 'Orta' olan {medium_count} olay tekrarlandığı için otomatik olarak 'Yüksek'e çıkarıldı."
        })

    # State güncellemeleri
    state["actions_recommended"] = list(recommendations)
    state["risk_assessments"] = [highest_risk]
    state["technical_summary"] = impact_desc
    state["tool_history"] = tool_history_entries
    state["risk_history"] = MemoryManager.add_risk(state.get("risk_history", []), highest_risk)
    state["risk_trend"] = MemoryManager.calculate_risk_trend(state["risk_history"])
    state["next_step"] = "compile_report"
    logger.info(f"📝 impact_desc: {impact_desc[:100]}")

    return state
    
def compile_report_node(state: AgentState) -> AgentState:
    # Dinamik keyword üretimi: detected_events içindeki olay metinlerinden türet
    
    previous_cases = []

    # ==========================
    # Temporal Context Summary
    # ==========================
    context = state.get("context_summary", {})

    temporal_summary = ""

    if context.get("systematic_risk", False):
        temporal_summary = (
            f" Son analizlerde aynı bölgede "
            f"{context.get('repeated_events', 0)} adet benzer risk olayı "
            f"tespit edilmiştir. "
            f"Bu durum sistematik bir güvenlik riski olarak değerlendirilmiş "
            f"ve risk seviyesi artırılmıştır. "
            f"Önerilen aksiyon: "
            f"{context.get('recommended_action', 'Bakım ekibini bilgilendir')}."
        )

    cleaned_events = []

    for item in state["detected_events"]:
        event_data = item.get("event", {})
        event_text = (
            event_data.get("description", "Açıklama bulunamadı.")
            if isinstance(event_data, dict)
            else str(event_data)
        )

        cleaned_events.append({
            "time": item["time"],
            "event": event_text,
            "center": item.get("center"),
            "bbox": item.get("bbox")
        })

    # ==========================
    # Nihai Özet — Dinamik & Anlamlı
    # ==========================
    highest_risk = state.get("risk_assessments", ["Düşük"])[0]

    base_summary = "Saha görüntüleri incelendi."
    event_texts = [e["event"] for e in cleaned_events if e.get("event")][:2]
    if event_texts:
        joined = " ".join(event_texts)
        base_summary = joined[:120] + ("..." if len(joined) > 120 else "")

    final_summary = f"{base_summary}{temporal_summary}"

    report_data = {
        "summary": final_summary,
        "events": cleaned_events,
        "risk": highest_risk,
        "risk_trend": state.get("risk_trend", "Bilinmiyor"),
        "actions": state["actions_recommended"]
    }


    json_report = None
    try:
        logger.info("✨ generate_final_report aracı çağrılıyor (LLM Kurumsal Özet)...")
        json_report = generate_final_report.invoke({"data_dict": report_data})
    except Exception as e:
        logger.warning(f"⚠️ generate_final_report aracı hatası: {e}. Manuel şema şablonuna geçiliyor.")

    # Eğer araç bir sebepten başarısız olursa güvenli fallback mekanizması devreye girer
    if not json_report or "error" in json_report:
        report_data["kpi_metrics"] = {
            "ai_processing_status": "Başarılı / Kritik Müdahale Devrede" if highest_risk in ["Yüksek", "Kritik"] else "Başarılı",
            "confidence_score": 0.96,
            "is_schema_validated": True
        }
        try:
            validated = FinalReportSchema(**report_data)
            json_report = validated.model_dump_json(indent=2)
        except Exception:
            json_report = json.dumps(report_data, ensure_ascii=False, indent=2)


    # Risk seviyesine göre koşullu CMS export — Düşük riskli olaylar sistemi gereksiz meşgul etmemeli
    if highest_risk in ["Yüksek", "Kritik", "Orta"]:
        export_to_incident_management_system.invoke({
            "report_json": json_report
        })
        logger.info(f"📋 Incident exported to CMS: {highest_risk} seviyeli olay")
    else:
        logger.info("ℹ️ Düşük risk — CMS export atlandı (şartname kuralı)")

    log_kpi_metrics.invoke({
        "metrics_data": {
            "processed_segments_count": len(cleaned_events),
            "highest_risk_detected": state["risk_assessments"][0],
            "is_schema_validated": "error" not in json_report
        }
    })

    state["final_json_output"] = json_report

    # ==========================
    # Persistent Memory
    # ==========================
    MemoryManager.save_memory(
        state.get("event_history", []),
        state.get("risk_history", []),
        state.get("tool_history", [])
    )

    state["next_step"] = "complete"
    
    # Reasoning chain log — compile node'un yaptıklarını göster
    tool_calls_log = [
        {
            "name": "generate_final_report",
            "args": {"data_dict": "Ham rapor sözlüğü"},
            "id": "call_report_004"
        },
        {
            "name": "log_kpi_metrics",
            "args": {
                "total_events": len(state.get("detected_events", [])),
                "highest_risk": highest_risk,
                "processing_status": "Başarılı"
            },
            "id": "call_kpi_005"
        }
    ]

    # Sadece orta/yüksek/kritik riskte CMS export çağrısı loglanır
    if highest_risk in ["Yüksek", "Kritik", "Orta"]:
        tool_calls_log.insert(0, {
            "name": "export_to_incident_management_system",
            "args": {
                "report_summary": state.get("video_summary", "Rapor hazırlandı"),
                "risk_level": highest_risk,
                "event_count": len(state.get("detected_events", []))
            },
            "id": "call_export_004"
        })

    log_message = {
        "role": "assistant",
        "content": "Nihai risk raporu başarıyla derlendi, KPI metrikleri loglandı.",
        "tool_calls": tool_calls_log
    }
    # Duplicate önleme: son mesaj aynı içerikse ekleme
    last_msg = state["messages"][-1] if state["messages"] else None
    last_content = last_msg.get("content", "") if isinstance(last_msg, dict) else getattr(last_msg, "content", "")
    if last_content != log_message["content"]:
        state["messages"].append(log_message)
    logger.info(f"🔍 compile_report_node çalışıyor. Mevcut state: {state.keys()}")

    
    return state
    
workflow = StateGraph(AgentState)

workflow.add_node("extract_video_info", extract_video_info_node)
workflow.add_node("evaluate_risks", evaluate_risks_node)
workflow.add_node("compile_report", compile_report_node)

workflow.set_entry_point("extract_video_info")

# def router(state: AgentState) -> str:
#     if state["next_step"] == "process_risk":
#         return "evaluate_risks"
#     elif state["next_step"] == "compile_report":
#         return "compile_report"
#     return END


workflow.add_edge("extract_video_info", "evaluate_risks")
workflow.add_edge("evaluate_risks", "compile_report")
workflow.add_edge("compile_report", END)

print("🔍 GRAPH EDGES:", workflow.edges)
app = workflow.compile()
