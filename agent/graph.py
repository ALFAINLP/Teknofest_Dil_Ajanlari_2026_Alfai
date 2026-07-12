from langgraph.graph import StateGraph, END
from agent.state import AgentState
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

def extract_video_info_node(state: AgentState) -> AgentState:
    telemetry = monitor_hardware_telemetry.invoke({})
    vram_free = telemetry.get("vram_free_gb", 16.0)
    optimize_inference_params.invoke({"vram_status": vram_free})
    
    video_path = state.get("video_path", "video_TEKNOFEST_2026.mp4")
    vision_output = state.get("vision_output")

    if vision_output is None:
        vision_output = analyze_video(video_path)
    scene_meta = get_scene_metadata.invoke({"video_path": video_path})
    zone = scene_meta.get("zone_name", "Genel Saha")
    
    critical_seconds = flag_critical_timestamps.invoke({"video_path": video_path})
    
    detected = []
    target_segments = sorted(list(set(state["current_time_segments"] + critical_seconds)))
    
    for t in target_segments:
    	motion = detect_motion_density.invoke({
        	"start_time": t,
        	"end_time": t + 2,
        	"video_path": video_path
    	})

    	frame_info = get_frame.invoke({
        	"time": t,
        	"video_path": video_path
    	})

    	minutes, seconds = divmod(int(t), 60)
        time_str = f"{minutes:02d}:{seconds:02d}"

    	detected.append({
        	"time": time_str,
        	"event": frame_info,
        	"motion_density": motion.get("average_motion_density_percentage", 0.0),
        	"zone": zone
    	})
    
    state["detected_events"] = detected
    state["next_step"] = "process_risk"
    return state

def evaluate_risks_node(state: AgentState) -> AgentState:
    highest_risk = "Düşük"
    recommendations = set()
    
    zone = state["detected_events"][0]["zone"] if state["detected_events"] else "Genel Saha"
    env_hazards = assess_environmental_hazards.invoke({"zone": zone})
    
    chronological_events_for_tool = []
    for item in state["detected_events"]:
        event_data = item.get("event", {})
        event_text = event_data.get("description", "Açıklama bulunamadı.") if isinstance(event_data, dict) else str(event_data)
        chronological_events_for_tool.append({
            "time": item["time"],
            "event": event_text
        })
    
    risk_analysis_result = analyze_risk.invoke({"chronological_events": chronological_events_for_tool})
    highest_risk = risk_analysis_result.get("overall_risk_level", "Düşük")
    
    if highest_risk == "Yüksek" or env_hazards.get("secondary_hazard_trigger", False):
        highest_risk = "Yüksek"
        
        personnel = get_personnel_on_duty.invoke({"timestamp": 15, "zone": zone})
        for person in personnel:
            if person.get("role") in ["Nöbetçi İş Yeri Hekimi", "Saha Güvenlik Şefi"]:
                recommendations.add(f"Bilgi Verilecek Sorumlu: {person['name']} ({person['role']}) -> Dahili: {person.get('contact_ext', 'N/A')}")
                
        recommendations.add("Sağlık ekibini çağır")
        recommendations.add("Alanı güvenlik altına al")
        
    elif highest_risk == "Orta":
        recommendations.add("Saha amirine haber ver")

    if highest_risk == "Yüksek":
        trigger_operator_alert.invoke({
            "alert_level": "YÜKSEK",
            "raw_incident_context": risk_analysis_result.get("impact_assessment", "")
        })

    state["actions_recommended"] = list(recommendations)
    state["risk_assessments"] = [highest_risk] 
    state["next_step"] = "compile_report"
    return state

def compile_report_node(state: AgentState) -> AgentState:
    previous_cases = query_previous_incidents.invoke({"incident_keywords": "forklift devrilmesi peron 3"})
    memory_summary = ""
    if previous_cases:
        memory_summary = f" [Hafıza Uyarısı: {previous_cases[0]['date']} tarihinde bu bölgede kök nedeni '{previous_cases[0]['root_cause']}' olan benzer bir vaka yaşanmıştır!]"

    cleaned_events = []
    for item in state["detected_events"]:
        event_data = item.get("event", {})
        event_text = event_data.get("description", "Açıklama bulunamadı.") if isinstance(event_data, dict) else str(event_data)
        cleaned_events.append({
            "time": item["time"],
            "event": event_text
        })

    report_data = {
        "summary": f"Videoda forklift kazası ve acil müdahale gerektiren yaralanma riski gözlenmiştir.{memory_summary}" if state["risk_assessments"][0] == "Yüksek" else "Rutin operasyon gözlenmiştir.",
        "events": cleaned_events,
        "risk": state["risk_assessments"][0],
        "actions": state["actions_recommended"]
    }
    
    json_report = generate_final_report(report_data)
    
    export_to_incident_management_system.invoke({"report_json": json_report})
    
    log_kpi_metrics.invoke({
        "metrics_data": {
            "processed_segments_count": len(cleaned_events),
            "highest_risk_detected": state["risk_assessments"][0],
            "is_schema_validated": "error" not in json_report
        }
    })
    
    state["final_json_output"] = json_report
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
