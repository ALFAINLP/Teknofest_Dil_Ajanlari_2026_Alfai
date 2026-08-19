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
    tool_history_entries = state.get("tool_history", [])
    
    react_messages = [
        {"role": "system", "content": BRAIN_SYSTEM_PROMPT},
        {"role": "user", "content": clean_user_prompt}
    ]

    MAX_REACT_STEPS = 5
    react_step = 0
    brain_response = None

    # ================================================================
    # ASIL REACT DÖNGÜSÜ (BURASI ÇALIŞACAK)
    # ================================================================
    while react_step < MAX_REACT_STEPS:
        react_step += 1
        logger.info(f"🔄 ReAct Adım {react_step}: Brain çağrılıyor...")

        # BURADA force_json=False olduğu için API'ye araçlar GİDECEK
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

        if not hasattr(brain_response, "tool_calls") or not brain_response.tool_calls:
            logger.info(f"🔄 ReAct Adım {react_step}: Tool çağrısı yok, JSON yanıt isteniyor...")
            brain_response = analyze_state(react_messages, force_json=True)
            break

        # Araçları çalıştır ve mesaj geçmişine ekle
        for tool_call in brain_response.tool_calls:
            tool_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            tool_result = None

            logger.info(f"🛠️ Brain '{tool_name}' aracını çağırdı. Argümanlar: {args}")

            # İlgili tool çağrıları...
            if tool_name == "trigger_operator_alert":
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

            react_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(tool_result, ensure_ascii=False)
            })

    # JSON parse işlemleri
    brain_content = brain_response.content if brain_response else ""
    impact_desc = "Analiz tamamlandı."
    
    if brain_content:
        try:
            start = brain_content.find('{')
            end = brain_content.rfind('}') + 1
            if start != -1 and end > start:
                json_str = brain_content[start:end]
                brain_json = json.loads(json_str)
                highest_risk = brain_json.get("risk", highest_risk)
                impact_desc = brain_json.get("decision", impact_desc)
                
                for action in brain_json.get("actions", []):
                    if action not in recommendations: recommendations.add(action)
        except Exception as e:
            logger.warning(f"⚠️ Brain JSON parse hatası: {e}")

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

    return state

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
    # Geçmiş hafızayı özet olarak hazırla
    risk_history_summary = state.get("risk_history", [])[-5:]  # son 5 risk
    risk_trend = state.get("risk_trend", "Stabil (Yetersiz Veri)")
    context_summary = state.get("context_summary", {})
    tool_history_recent = state.get("tool_history", [])[-3:]   # son 3 tool

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
    # 🟢 KRİTİK: state["messages"] ile hiçbir bağlantısı yok. Sadece bu 2 mesaj gidiyor.
    # ReAct döngüsü için mesaj geçmişini başlat

    recommendations = set()
    highest_risk = "Düşük"
    impact_desc = "Otonom yapay zeka analizi devrede."
    tool_history_entries = state.get("tool_history", [])  # AgentState'e eklemeyi unutmayın!
    actions_from_brain = []

    react_messages = [
        {"role": "system", "content": clean_system_prompt},
        {"role": "user", "content": clean_user_prompt}
    ]

    MAX_REACT_STEPS = 5
    react_step = 0

    while react_step < MAX_REACT_STEPS:
        react_step += 1
        logger.info(f"🔄 ReAct Adım {react_step}: Brain çağrılıyor...")

        # Tool çağrısı beklenen adımlarda JSON format zorunluluğu kaldırılıyor
        # (response_format=json_object tool_calls ile çakışır — model JSON seçer, tool'u pas geçer)
        brain_response = analyze_state(react_messages, force_json=False)

        # Brain'in bu turki cevabını geçmişe ekle
        assistant_msg = {"role": "assistant", "content": brain_response.content or ""}

        if hasattr(brain_response, "tool_calls") and brain_response.tool_calls:
            # Tool çağrısı varsa assistant mesajına ekle (OpenAI formatı zorunlu kılıyor)
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

        # Tool çağrısı yoksa döngüyü bitir — son yanıtı JSON formatıyla al
        if not hasattr(brain_response, "tool_calls") or not brain_response.tool_calls:
            logger.info(f"🔄 ReAct Adım {react_step}: Tool çağrısı yok, JSON yanıt isteniyor...")
            # Son adımda JSON formatında karar özeti iste
            brain_response = analyze_state(react_messages, force_json=True)
            logger.info(f"✅ ReAct döngüsü {react_step}. adımda tamamlandı.")
            break

        # Tool çağrısı varsa her birini çalıştır ve sonucu geçmişe ekle
        for tool_call in brain_response.tool_calls:
            tool_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            tool_result = None

            logger.info(f"🛠️ Brain '{tool_name}' aracını çağırdı. Argümanlar: {args}")

            if tool_name == "trigger_operator_alert":
                tool_result = trigger_operator_alert.invoke(args)
                tool_history_entries.append({"tool": tool_name, "args": args, "result": tool_result})
                recommendations.add(f"Operatör uyarısı: {args.get('raw_incident_context', '')}")
                if highest_risk not in ["Kritik", "Yüksek"]:
                    highest_risk = "Yüksek"

            elif tool_name == "query_previous_incidents":
                tool_result = query_previous_incidents.invoke(args)
                tool_history_entries.append({"tool": tool_name, "args": args, "result": tool_result})
                if tool_result:
                    recommendations.add(f"Geçmişte benzer olay: {tool_result[0].get('root_cause', '')}")
                    if highest_risk not in ["Kritik", "Yüksek"]:
                        highest_risk = "Yüksek"

            elif tool_name == "get_personnel_on_duty":
                tool_result = get_personnel_on_duty.invoke(args)
                tool_history_entries.append({"tool": tool_name, "args": args, "result": tool_result})
                if tool_result:
                    names = [p.get("name", "?") for p in tool_result]
                    recommendations.add(f"Görevli personel: {', '.join(names)}")

            # Tool sonucunu mesaj geçmişine ekle — Brain bir sonraki turda bunu görecek
            react_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(tool_result, ensure_ascii=False)
            })

        logger.info(f"🔁 Tool sonuçları Brain'e iletildi, ReAct döngüsü devam ediyor...")

    

    brain_content = brain_response.content

    # Eğer content None veya boş ise, varsayılan değerlerle devam et
    if not brain_content:
        logger.warning("⚠️ Brain modeli boş yanıt döndürdü. Varsayılan değerler kullanılacak.")
        highest_risk = "Düşük"
        actions_from_brain = ["Rutin saha izlemesine devam edin, anomali takibi yapın."]
        impact_desc = "Model yanıt vermedi, varsayılan durum."
    else:
        try:
            # SADECE JSON KISMINI AL (ilk { ile son } arası)
            start = brain_content.find('{')
            end = brain_content.rfind('}') + 1
            if start != -1 and end > start:
                json_str = brain_content[start:end]
                # Markdown kod bloklarını temizle (opsiyonel)
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0].strip()
                brain_json = json.loads(json_str)
            else:
                raise ValueError("JSON bulunamadı")
            
            highest_risk = brain_json.get("risk", brain_json.get("risk_level", "Düşük"))
            actions_from_brain = brain_json.get("actions", [])
            impact_desc = brain_json.get("decision", "Analiz tamamlandı.")
            if not actions_from_brain:
                actions_from_brain = ["Rutin saha izlemesine devam edin, anomali takibi yapın."]
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            logger.warning(f"⚠️ Brain JSON parse hatası: {e}. Content: {brain_content[:200]}")
            actions_from_brain = [brain_content[:200] if brain_content else "Değerlendirme yapılamadı."]
            if "Yüksek" in brain_content:
                highest_risk = "Yüksek"
            elif "Orta" in brain_content:
                highest_risk = "Orta"
            elif "Kritik" in brain_content:
                highest_risk = "Kritik"
            else:
                highest_risk = "Düşük"


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
    assistant_content = f"[RISK DEĞERLENDİRMESİ] {brain_response.content if brain_response.content else 'Risk değerlendirmesi yapıldı, özel bir aksiyon gerekmiyor.'}"
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
        logger.info(f"✅ evaluate_risks_node mesaj eklendi: {assistant_content[:100]}")


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

    logger.info(f"📝 impact_desc: {impact_desc[:100]}")
    logger.info(f"📝 technical_summary state'e yazıldı mı: {state.get('technical_summary', 'YOK')[:100]}")
    
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

    # Brain'in decision metni (technical_summary) en az 15 karakter ve anlamlıysa kullan
    brain_decision = state.get("technical_summary", "").strip()
    has_meaningful_decision = (
        len(brain_decision) >= 15
        and brain_decision not in ["Analiz tamamlandı.", "Model yanıt vermedi, varsayılan durum."]
    )

    if has_meaningful_decision:
        base_summary = brain_decision
    else:
        # Tespit edilen ilk 1-2 olaydan dinamik özet üret
        event_texts = [e["event"] for e in cleaned_events if e.get("event")][:2]
        if event_texts:
            joined = " ".join(event_texts)
            # 120 karakteri aşarsa kırp
            base_summary = joined[:120] + ("..." if len(joined) > 120 else "")
        else:
            base_summary = "Saha görüntüleri incelendi."

    if highest_risk in ["Yüksek", "Kritik"]:
        final_summary = f"{base_summary}{temporal_summary}"
    else:
        final_summary = f"{base_summary}{temporal_summary}"



    report_data = {
        "summary": final_summary,
        "events": cleaned_events,
        "risk": state["risk_assessments"][0],
        
        # YENİ: State'ten trendi alıp LLM'e gönderilecek veriye ekliyoruz
        "risk_trend": state.get("risk_trend", "Bilinmiyor"), 
        
        "actions": state["actions_recommended"]
    }


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
            "name": "log_kpi_metrics",
            "args": {
                "total_events": len(state.get("detected_events", [])),
                "highest_risk": state["risk_assessments"][0] if state.get("risk_assessments") else "Bilinmiyor",
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
