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

def extract_video_info_node(state: AgentState) -> AgentState:
    telemetry = monitor_hardware_telemetry.invoke({})
    vram_free = telemetry.get("vram_free_gb", 16.0)
    optimize_inference_params.invoke({"vram_status": vram_free})
    
    scene_meta = get_scene_metadata.invoke({"video_path": state["video_path"]})
    zone = scene_meta.get("zone_name", "Genel Saha")
    
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
            "end_time": t + 2,
            "video_path": state["video_path"]
        })
        frame_info = get_frame.invoke({
            "time": t,
            "video_path": state["video_path"]
        })

        minutes = t // 60
        seconds = t % 60
        time_str = f"{minutes:02d}:{seconds:02d}"

        # FIX 1: get_frame bir dict döndürür; ham dict yerine description string'ini al
        if isinstance(frame_info, dict):
            event_text = frame_info.get("description", "Kare analizi tamamlandı.")
        else:
            event_text = str(frame_info)

        detected.append({
            "time": time_str,
            "event": event_text,
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
    # 1. BİLGİ TOPLAYAN MOCK FONKSİYONLAR (LLM'e veri sağlamak için)
    scene_meta = get_scene_metadata.invoke({"video_path": state["video_path"]})
    zone = scene_meta.get("zone_name", "Genel Saha")
    env_hazards = assess_environmental_hazards.invoke({"zone": zone})

    # FIX 2: event içeriği her zaman string olarak temizle (dict sızmasını önle)
    events_for_brain = []
    for item in state["detected_events"]:
        raw = item.get("event", "Açıklama bulunamadı.")
        if isinstance(raw, dict):
            event_text = raw.get("description", str(raw))
        else:
            event_text = str(raw)
        events_for_brain.append(f"Zaman: {item['time']} - Olay: {event_text}")

    # Llama'ya durum bilgisini ve araç (tool) kullanma yetkisini veriyoruz
    prompt = (
        f"Şu anki bölge: {zone}. Çevresel riskler: {env_hazards}. Aşağıdaki olayları incele. "
        "Eğer yüksek bir risk (örneğin kaza, devrilme) varsa 'trigger_operator_alert' "
        "aracını kullan ve önlemler al. Yoksa sadece risk seviyesini belirt.\n"
        f"Olaylar:\n{chr(10).join(events_for_brain)}"
    )

    # FIX 3: add_messages reducer çift yazma sorunu — state listesini mutate etme.
    # Önceki mesajları kopyala, yeni user prompt'unu ekle → analyze_state'e gönder.
    # State'e sadece YENİ mesajları döndür; reducer otomatik birleştirir.
    previous_messages = list(state.get("messages", []))
    messages_for_llm = previous_messages + [{"role": "user", "content": prompt}]

    # 2. LLM DÜŞÜNÜYOR VE KARAR VERİYOR
    brain_response = analyze_state(messages_for_llm)

    recommendations = set()
    highest_risk = "Düşük"
    impact_desc = "Otonom yapay zeka analizi devrede."

    # 3. AKSİYON ALAN MOCK FONKSİYONLAR (LLM inisiyatif alırsa çalışır)
    tool_called = False
    if hasattr(brain_response, "tool_calls") and brain_response.tool_calls:
        for tool_call in brain_response.tool_calls:
            if tool_call.function.name == "trigger_operator_alert":
                highest_risk = "Yüksek"
                args = json.loads(tool_call.function.arguments)
                trigger_operator_alert.invoke(args)
                recommendations.add(f"LLM Uyarı Aracı Tetikledi: {args}")
                impact_desc = args.get("raw_incident_context", "Kritik Olay Tespiti")
                tool_called = True
                print(f"🚨 [LLAMA TOOL CALL] trigger_operator_alert tetiklendi → {args}")

    # LLM yanıtını log mesajı olarak oluştur
    if tool_called:
        assistant_content = f"⚡ trigger_operator_alert çalıştırıldı. Risk: Yüksek. Bağlam: {impact_desc}"
    elif brain_response.content:
        assistant_content = brain_response.content
    else:
        assistant_content = "Llama değerlendirdi — araç çağrısı gerek görülmedi."

    # FIX 3 devam: Sadece bu adımda üretilen YENİ mesajları state'e yaz
    new_messages = [
        {"role": "user",      "content": prompt},
        {"role": "assistant", "content": assistant_content},
    ]
    state["messages"] = new_messages  # add_messages reducer bunları öncekilerle birleştirir

    state["actions_recommended"] = list(recommendations)
    state["risk_assessments"] = [highest_risk]
    state["technical_summary"] = impact_desc

    # ==========================
    # Adaptive Risk Memory
    # ==========================
    medium_count = MemoryManager.recent_risk_count(
        state.get("risk_history", []),
        "Orta"
    )
    if highest_risk == "Orta" and medium_count >= 2:
        highest_risk = "Yüksek"

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
    
    # FIX 3 devam: compile_report da yalnızca kendi yeni mesajını döndürür
    log_message = {
        "role": "assistant",
        "content": "Nihai risk raporu başarıyla derlendi, CMS'e aktarıldı ve KPI metrikleri loglandı.",
        "tool_calls": [
            {
                "name": "export_to_incident_management_system",
                "args": {},
                "id": "call_export_004"
            },
            {
                "name": "log_kpi_metrics",
                "args": {},
                "id": "call_kpi_005"
            }
        ]
    }
    state["messages"] = [log_message]  # reducer öncekilerle birleştirir


    

    return state
    
workflow = StateGraph(AgentState)

workflow.add_node("extract_video_info", extract_video_info_node)
workflow.add_node("evaluate_risks", evaluate_risks_node)
workflow.add_node("compile_report", compile_report_node)

workflow.set_entry_point("extract_video_info")

def router(state: AgentState) -> str:
    if state["next_step"] == "process_risk":
        return "evaluate_risks"
    elif state["next_step"] == "compile_report":
        return "compile_report"
    return END


workflow.add_conditional_edges("extract_video_info", router)
workflow.add_conditional_edges("evaluate_risks", router)
workflow.add_edge("compile_report", END)

app = workflow.compile()
