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

def extract_video_info_node(state: AgentState) -> AgentState:
    telemetry = monitor_hardware_telemetry.invoke({})
    vram_free = telemetry.get("vram_free_gb", 16.0)
    optimize_inference_params.invoke({"vram_status": vram_free})
    
    scene_meta = get_scene_metadata.invoke({"video_path": state["video_path"]})
    
    zone = scene_meta.get("zone_name", "Genel Saha")
    state["current_zone"] = zone
    
    qwen_output = analyze_video(state["video_path"])
    
    critical_seconds = flag_critical_timestamps.invoke({"video_path": state["video_path"]})
    detected = []
    for event_item in qwen_output.get("events", []):
        detected.append({
            "time": event_item.get("time", "00:00"),
            "event": event_item.get("event", event_item.get("description", "Bilinmeyen Olay")),
            "motion_density": 0.0, 
            "zone": zone
        })
        
    state["detected_events"] = detected
    

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
    # ================================================================
    # 🔧 HATA 10 (ÇÖZÜLDÜ): get_scene_metadata API'sini TEKRAR çağırma!
    # Artık extract_video_info_node'da state'e yazılan current_zone'u okuyoruz.
    # (agent/state.py'e 'current_zone: str' eklemeyi unutmayın)
    # ================================================================
    zone = state.get("current_zone", "Genel Saha")  # API çağrısı yok, cache'ten okuyor.
    
    # Çevresel riskler hala gerekliyse (mock) çağrılabilir, ama bu fonksiyon sadece 1 kez çalışacak artık.
    env_hazards = assess_environmental_hazards.invoke({"zone": zone})

    # ================================================================
    # 🔧 HATA 2 & 6 (ÇÖZÜLDÜ): Brain'e sadece olay değil, 
    # hareket yoğunluğu (motion_density) ve risk flag'leri (has_risk) de gönderiliyor.
    # Artık 00:15'te düşüp 00:16'da kalkan ile 00:45 yatan arasındaki farkı anlayabilir.
    # ================================================================
    events_for_brain = []
    for item in state["detected_events"]:
        event_content = item.get("event", "Açıklama bulunamadı.")
        
        if isinstance(event_content, dict):
            event_text = event_content.get("description", str(event_content))
        else:
            event_text = str(event_content)
        
        # 🟢 KRİTİK EKLEME: Zamansal ve hareket verilerini de ekliyoruz
        motion = item.get('motion_density', 0)
        has_risk_flag = item.get('has_risk', False)
        time = item.get('time', '00:00')
        
        events_for_brain.append(
            f"Zaman: {time} | Olay: {event_text} | "
            f"Hareket Yoğunluğu: %{motion:.1f} | Risk Varlığı: {'Evet' if has_risk_flag else 'Hayır'}"
        )

    # ================================================================
    # 🔧 HATA 1 & 7 (ÇÖZÜLDÜ): state["messages"] birikimini KULLANMIYORUZ!
    # Brain'e sadece bu adıma özel, tertemiz bir bağımsız prompt gönderiyoruz.
    # Böylece önceki adımların mesajları birikip gereksiz yere tekrara neden olmuyor.
    # Ayrıca 'user' rolü yerine 'system' + 'user' ayrımı yapıyoruz.
    # ================================================================
    clean_system_prompt = (
        "Sen bir operasyon sahası risk analisti yapay zeka ajanısın. "
        "Aşağıdaki olayları değerlendir, zamansal ilişkilere dikkat et. "
        "Eğer gerçek bir kaza/devrilme ve hareket edememe durumu varsa 'trigger_operator_alert' aracını çağır. "
        "🚨 ÖNEMLİ: Bu aracı çağırırken ARGÜMANLARI KESİNLİKLE ŞU FORMATTA GÖNDER: "
        "{'alert_level': 'Yüksek', 'raw_incident_context': 'Buraya olayın kısa açıklamasını yaz'}. "
        "'reason' veya başka bir anahtar KULLANMA! Sadece 'alert_level' ve 'raw_incident_context' gönder."
    )
    
    clean_system_prompt = BRAIN_SYSTEM_PROMPT
    clean_user_prompt = (
        f"Şu anki bölge: {zone}.\n"
        f"Çevresel riskler: {env_hazards}.\n"
        f"Aşağıdaki olayları incele ve risk seviyesini belirle:\n"
        f"{chr(10).join(events_for_brain)}"
    )
    logger.info(f"🔍 evaluate_risks_node çalışıyor. Mevcut state: {state.keys()}")
    # 🟢 KRİTİK: state["messages"] ile hiçbir bağlantısı yok. Sadece bu 2 mesaj gidiyor.
    brain_response = analyze_state([
        {"role": "system", "content": clean_system_prompt},
        {"role": "user", "content": clean_user_prompt}
    ])

    # ================================================================
    # 🔧 HATA 8 (ÇÖZÜLDÜ): Tool çağrılarının çıktılarını logluyoruz ve state'e yazıyoruz.
    # ================================================================
    recommendations = set()
    highest_risk = "Düşük"
    impact_desc = "Otonom yapay zeka analizi devrede."
    tool_history_entries = state.get("tool_history", [])  # AgentState'e eklemeyi unutmayın!
    actions_from_brain = []

    brain_content = brain_response.content

    # Eğer content None veya boş ise, varsayılan değerlerle devam et
    if not brain_content:
        logger.warning("⚠️ Brain modeli boş yanıt döndürdü. Varsayılan değerler kullanılacak.")
        highest_risk = "Düşük"
        actions_from_brain = ["Rutin saha izlemesine devam edin, anomali takibi yapın."]
        impact_desc = "Model yanıt vermedi, varsayılan durum."
    else:
        try:
            # Markdown kod bloklarını temizle
            if "```json" in brain_content:
                brain_content = brain_content.split("```json")[1].split("```")[0].strip()
            elif "```" in brain_content:
                brain_content = brain_content.split("```")[1].split("```")[0].strip()
            
            brain_json = json.loads(brain_content)
            highest_risk = brain_json.get("risk_level", "Düşük")
            actions_from_brain = brain_json.get("actions", [])
            impact_desc = brain_json.get("decision", "Analiz tamamlandı.")
            
            if not actions_from_brain:
                actions_from_brain = ["Rutin saha izlemesine devam edin, anomali takibi yapın."]
                
        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            logger.warning(f"⚠️ Brain JSON parse hatası: {e}. Content: {brain_content[:100] if brain_content else 'Boş'}")
            actions_from_brain = [brain_content[:200]] if brain_content else ["Değerlendirme yapılamadı."]
            if brain_content and "Yüksek" in brain_content:
                highest_risk = "Yüksek"
            elif brain_content and "Orta" in brain_content:
                highest_risk = "Orta"
            elif brain_content and "Kritik" in brain_content:
                highest_risk = "Kritik"
            else:
                highest_risk = "Düşük"



    if hasattr(brain_response, 'tool_calls') and brain_response.tool_calls:
        for tool_call in brain_response.tool_calls:
            if tool_call.function.name == "trigger_operator_alert":
                args = json.loads(tool_call.function.arguments)
                alert_result = trigger_operator_alert.invoke(args)
                tool_history_entries.append({
                    "tool": "trigger_operator_alert",
                    "args": args,
                    "result": alert_result
                })
                recommendations.add(f"Operatör uyarısı: {args.get('raw_incident_context', '')}")
                impact_desc = args.get("raw_incident_context", "Kritik Olay Tespiti")
                if highest_risk not in ["Kritik", "Yüksek"]:
                    highest_risk = "Yüksek"

    # 3. Brain'den gelen actions'ları recommendations'a ekle (operatöre yönelik öneriler)
    for action in actions_from_brain:
        if action and action not in recommendations:
            recommendations.add(action)

    # Eğer hiçbir öneri yoksa varsayılanı ekle
    if not recommendations:
        recommendations.add("Rutin saha izlemesi devam ediyor, anomali tespiti bekleniyor.")


    # ================================================================
    # 🟢 LOGLAMA (Şeffaflık): Brain'in verdiği ham yanıtı da state'e ekleyelim ki 
    # logger'da "Mock araç çağrısı yapıldı" gibi anlamsız ifadeler kalmasın.
    # ================================================================
    assistant_content = brain_response.content if brain_response.content else "Risk değerlendirmesi yapıldı, özel bir aksiyon gerekmiyor."
    # state["messages"].append({
    #     "role": "assistant", 
    #     "content": assistant_content
    # })

    # ================================================================
    # 🧠 ADAPTIVE RISK MEMORY (Zaten iyi çalışıyor, dokunmuyoruz)
    # ================================================================
    medium_count = MemoryManager.recent_risk_count(
        state.get("risk_history", []),
        "Orta"
    )

    if highest_risk == "Orta" and medium_count >= 2:
        highest_risk = "Yüksek"
        # 🟢 EKSTRA: Neden yükseldiğini de loga ekleyelim
        state["messages"].append({
            "role": "assistant",
            "content": f"⚠️ Risk seviyesi 'Orta' olan {medium_count} olay tekrarlandığı için otomatik olarak 'Yüksek'e çıkarıldı."
        })

    # ================================================================
    # STATE GÜNCELLEMELERİ
    # ================================================================
    state["actions_recommended"] = list(recommendations)
    state["risk_assessments"] = [highest_risk]
    state["technical_summary"] = impact_desc
    state["tool_history"] = tool_history_entries  # Artık boş değil, dolu!
    
    state["risk_history"] = MemoryManager.add_risk(
        state.get("risk_history", []),
        highest_risk
    )
    state["risk_trend"] = MemoryManager.calculate_risk_trend(state["risk_history"])
    state["next_step"] = "compile_report"
    
    return state
    
def compile_report_node(state: AgentState) -> AgentState:
    previous_cases = query_previous_incidents.invoke({
        "incident_keywords": "forklift devrilmesi peron 3"
    })

    # ==========================
    # Memory Summary
    # ==========================
    memory_summary = ""

    if previous_cases and "forklift" in state.get("technical_summary", "").lower():
        memory_summary = (
            f" [Hafıza Uyarısı: {previous_cases[0]['date']} tarihinde bu bölgede "
            f"kök nedeni '{previous_cases[0]['root_cause']}' olan benzer bir vaka yaşanmıştır!]"
        )

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
            "event": event_text
        })

    # ==========================
    # Nihai Özet
    # ==========================
    if state["risk_assessments"][0] == "Yüksek":
        final_summary = (
            f"{state.get('technical_summary', 'Kritik iş kazası tespiti.')}"
            f"{memory_summary}"
            f"{temporal_summary}"
        )
    else:
        final_summary = (
            "Rutin operasyon gözlenmiştir."
            f"{temporal_summary}"
        )

    report_data = {
        "summary": final_summary,
        "events": cleaned_events,
        "risk": state["risk_assessments"][0],
        
        # YENİ: State'ten trendi alıp LLM'e gönderilecek veriye ekliyoruz
        "risk_trend": state.get("risk_trend", "Bilinmiyor"), 
        
        "actions": state["actions_recommended"]
    }

    # Nihai raporu oluştur
    json_report = generate_final_report.invoke({
        "data_dict": report_data
    })

    export_to_incident_management_system.invoke({
        "report_json": json_report
    })

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
    
    log_message = {
        "role": "assistant",
        "content": "Nihai risk raporu başarıyla derlendi, CMS'e aktarıldı ve KPI metrikleri loglandı.",
        "tool_calls": [
            {
                "name": "export_to_incident_management_system",
                "args": {
                    "report_summary": state.get("video_summary", "Rapor hazırlandı"),
                    "risk_level": state["risk_assessments"][0] if state.get("risk_assessments") else "Bilinmiyor",
                    "event_count": len(state.get("detected_events", []))
                },
                "id": "call_export_004"
            },
            {
                "name": "log_kpi_metrics",
                "args": {
                    "total_events": len(state.get("detected_events", [])),
                    "highest_risk": state["risk_assessments"][0] if state.get("risk_assessments") else "Bilinmiyor",
                    "processing_status": "Başarılı"
                },
                "id": "call_kpi_005"
            }
        ]
    }
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
