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

def extract_video_info_node(state: AgentState) -> AgentState:
    telemetry = monitor_hardware_telemetry.invoke({})
    vram_free = telemetry.get("vram_free_gb", 16.0)
    optimize_inference_params.invoke({"vram_status": vram_free})
    
    scene_meta = get_scene_metadata.invoke({"video_path": state["video_path"]})
    zone = scene_meta.get("zone_name", "Genel Saha")
    
    critical_seconds = flag_critical_timestamps.invoke({"video_path": state["video_path"]})
    detected = []
    target_segments = sorted(list(set(state["current_time_segments"] + critical_seconds)))
    
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
    return state

def evaluate_risks_node(state: AgentState) -> AgentState:
    highest_risk = "Düşük"
    recommendations = set()

    scene_meta = get_scene_metadata.invoke({"video_path": state["video_path"]})
    zone = scene_meta.get("zone_name", "Genel Saha")
    env_hazards = assess_environmental_hazards.invoke({"zone": zone})

    events_for_brain = []
    for item in state["detected_events"]:
        event_data = item.get("event", {})
        event_text = (
            event_data.get("description", "Açıklama bulunamadı.")
            if isinstance(event_data, dict)
            else str(event_data)
        )

        events_for_brain.append({
            "time": item["time"],
            "event": event_text
        })

    risk_analysis_result = analyze_risk.invoke({
        "chronological_events": events_for_brain
    })

    highest_risk = risk_analysis_result.get(
        "overall_risk_level",
        "Düşük"
    )

    impact_desc = risk_analysis_result.get(
        "impact_assessment",
        "Kritik durum tespiti."
    )

    # ==========================
    # Adaptive Risk Memory
    # ==========================
    medium_count = MemoryManager.recent_risk_count(
        state.get("risk_history", []),
        "Orta"
    )

    if highest_risk == "Orta" and medium_count >= 2:
        highest_risk = "Yüksek"

    # ==========================
    # Temporal Context
    # ==========================
    context = state.get("context_summary", {})

    if context.get("systematic_risk", False):

        if highest_risk == "Orta":
            highest_risk = "Yüksek"

        recommendations.add(
            "Temporal Context: Aynı bölgede tekrar eden olaylar tespit edildi."
        )

        recommendations.add(
            context.get(
                "recommended_action",
                "Bakım ekibini bilgilendir."
            )
        )

    # ==========================
    # Risk Değerlendirmesi
    # ==========================
    if highest_risk == "Yüksek" or env_hazards.get(
        "secondary_hazard_trigger",
        False
    ):
        highest_risk = "Yüksek"

        incident_type = (
            "forklift_kazasi"
            if "forklift" in impact_desc.lower()
            else "agir_yuk_ve_malzeme_dusmesi"
        )

        recommendations.add("Alanı güvenlik altına al")
        recommendations.add("Sağlık ekibini çağır")
        recommendations.add("Geçiş yollarını boşalt")

        if "forklift" in impact_desc.lower():
            recommendations.add(
                "Forklift hız limitlerini kontrol et"
            )
        else:
            recommendations.add(
                "Yükleme/indirme operasyon alanını askıya al ve tonaj sınırlarını denetle"
            )

        personnel = get_personnel_on_duty.invoke({
            "timestamp": 15,
            "zone": zone
        })

        for person in personnel:
            if person["role"] in [
                "Nöbetçi İş Yeri Hekimi",
                "Saha Güvenlik Şefi"
            ]:
                recommendations.add(
                    f"Bilgi Verilecek Sorumlu: "
                    f"{person['name']} "
                    f"({person['role']}) "
                    f"-> Dahili: {person.get('contact_ext', 'N/A')}"
                )

    elif highest_risk == "Orta":
        recommendations.add("Saha amirine haber ver")

    # ==========================
    # Tool Memory + Debouncing
    # ==========================
    if highest_risk == "Yüksek":

        if not MemoryManager.was_tool_used(
            state.get("tool_history", []),
            "trigger_operator_alert"
        ):

            trigger_operator_alert.invoke({
                "alert_level": "YÜKSEK",
                "raw_incident_context": impact_desc
            })

            state["tool_history"] = MemoryManager.add_tool(
                state.get("tool_history", []),
                "trigger_operator_alert"
            )

    # ==========================
    # State Güncellemeleri
    # ==========================
    state["actions_recommended"] = list(recommendations)
    state["risk_assessments"] = [highest_risk]

    state["risk_history"] = MemoryManager.add_risk(
        state.get("risk_history", []),
        highest_risk
    )

    state["technical_summary"] = impact_desc

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
