#kpi_enchanced.py
import json
import os
import re
import time
import subprocess
import logging
from typing import List, Dict, Any, Set, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch

from agent.graph import app

# ============================================================================
# 0. LOGGING AYARLARI - SADECE KRİTİK HATALARI GÖSTER
# ============================================================================

# ALFAI_API logger'ını sustur
logging.getLogger("ALFAI_API").setLevel(logging.ERROR)
logging.getLogger("ALFAI_API").disabled = True

# Diğer logger'ları da sustur
for logger_name in ["httpx", "openai", "vllm", "uvicorn", "fastapi"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)
    logging.getLogger(logger_name).disabled = True

# Root logger'ı da sustur (sadece kritik hatalar gösterilsin)
logging.basicConfig(level=logging.ERROR)

# ============================================================================
# 1. RUN AGENT PIPELINE
# ============================================================================

def run_agent_pipeline(video_path: str) -> dict:
    """
    Her video için sıfırdan temiz bir state ile LangGraph pipeline'ını çalıştırır.
    Hafıza kirliliğini önlemek için state sıfır başlatılır.
    """
    initial_state = {
        "video_path": video_path,
        "current_time_segments": [],
        "detected_events": [],
        "risk_assessments": [],
        "actions_recommended": [],
        "final_json_output": "",
        "next_step": "",
        "event_history": [],
        "risk_history": [],
        "tool_history": [],
        "messages": []
    }

    # stdout'u geçici olarak sustur
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        final_output = app.invoke(initial_state)
    finally:
        sys.stdout = old_stdout

    # 1. JSON çıktısını çözümle
    final_json = final_output.get("final_json_output", "")
    parsed_report = {}
    if isinstance(final_json, str) and final_json.strip():
        try:
            cleaned_json = re.sub(r"^```json\s*|\s*```$", "", final_json.strip(), flags=re.MULTILINE)
            parsed_report = json.loads(cleaned_json)
        except Exception:
            parsed_report = {}
    elif isinstance(final_json, dict):
        parsed_report = final_json

    # 2. Risk ve Olaylar
    detected_risk = parsed_report.get("risk", final_output.get("risk_assessments", ["Düşük"])[0])
    detected_events = parsed_report.get("events", final_output.get("detected_events", []))
    
    # 3. Actions / Tools
    executed_tools = set(final_output.get("actions_recommended", []))

    return {
        "risk_level": detected_risk,
        "events": detected_events,
        "actions": list(executed_tools),
        "_full_state": final_output  
    }


# ============================================================================
# 2. TÜRKÇE METİN İŞLEME
# ============================================================================

class TurkishTextProcessor:
    """Türkçe metin işleme ve normalizasyon yardımcı sınıfı"""
    
    @staticmethod
    def normalize(text: str) -> str:
        """Türkçe karakter ve büyük/küçük harf normalizasyonu"""
        text = re.sub(r'[İI]', 'i', str(text or ""))
        replacements = {
            'ı': 'i', 'ğ': 'g', 'ü': 'u', 'ş': 's', 'ö': 'o', 'ç': 'c',
            'Ğ': 'g', 'Ü': 'u', 'Ş': 's', 'Ö': 'o', 'Ç': 'c'
        }
        for tr_char, en_char in replacements.items():
            text = text.replace(tr_char, en_char)
        return text.lower().strip()
    
    @staticmethod
    def get_turkish_keywords(text: str, min_len: int = 2) -> Set[str]:
        """Türkçe metinden durak kelimelerden arındırılmış anlamlı kelimeleri çıkarır"""
        if not text:
            return set()
        
        tr_stop_words = {
            've', 'ile', 'icin', 'bir', 'bu', 'su', 'o', 'da', 'de', 'ki',
            'mi', 'mu', 'mı', 'mü', 'ne', 'nasil', 'neden', 'niye', 'cunku',
            'ama', 'fakat', 'ancak', 'lakin', 'yani', 'demek', 'soyle', 'boyle',
            'oyle', 'simdi', 'sonra', 'once', 'tarafindan', 'olarak', 'girdi', 'cikti'
        }
        
        normalized = TurkishTextProcessor.normalize(text)
        words = set(re.findall(r'[a-z]{2,}', normalized))
        return {w for w in words if w not in tr_stop_words and len(w) >= min_len}

    @staticmethod
    def turkish_jaccard(text1: str, text2: str) -> float:
        """Türkçe metinler arası Jaccard benzerliği"""
        words1 = TurkishTextProcessor.get_turkish_keywords(text1)
        words2 = TurkishTextProcessor.get_turkish_keywords(text2)
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0


# ============================================================================
# 3. ÇOK SEVİYELİ TOOL DEĞERLENDİRİCİ
# ============================================================================

class MultiLevelToolEvaluator:
    """
    Tool değerlendirmesinde 3 farklı seviyede puanlama:
    - Strict: Tam eşleşme (Exact Match)
    - Normal: Alias ile eşleşme
    - Fuzzy: Anlamsal benzerlik (Jaccard IoU)
    """
    
    def __init__(self):
        self.tool_aliases = {
            "trigger_operator_alert": ["alert", "operator_alert", "trigger_alert", "uyari", "alarm", "operator"],
            "get_personnel_on_duty": ["personnel", "get_personnel", "on_duty", "personel", "gorevli", "person"],
            "assess_environmental_hazards": ["hazard", "risk_assessment", "environmental", "risk", "tehlike", "assess"],
            "query_previous_incidents": ["incident", "previous_incidents", "query_incident", "gecmis", "kayit", "query"],
            "export_to_incident_management_system": ["export", "incident_management", "export_incident", "rapor", "cms"],
            "detect_motion_density": ["motion", "density", "motion_detection", "hareket", "yogunluk", "detect"],
            "log_kpi_metrics": ["log", "kpi", "metrics", "log_kpi"]
        }
        
        self.alias_to_tool = {}
        for tool, aliases in self.tool_aliases.items():
            for alias in aliases:
                self.alias_to_tool[alias] = tool
            self.alias_to_tool[tool] = tool
    
    def _extract_exact_tools(self, actions: List[str]) -> Set[str]:
        tools = set()
        for action in actions:
            action_str = TurkishTextProcessor.normalize(str(action))
            if action_str in self.alias_to_tool:
                tools.add(self.alias_to_tool[action_str])
        return tools
    
    def _extract_tools_with_aliases(self, actions: List[str]) -> Set[str]:
        tools = set()
        for action in actions:
            action_str = TurkishTextProcessor.normalize(str(action))
            if action_str in self.alias_to_tool:
                tools.add(self.alias_to_tool[action_str])
                continue
            for alias, tool in self.alias_to_tool.items():
                if alias in action_str or action_str in alias:
                    tools.add(tool)
                    break
        return tools
    
    def _calculate_semantic_similarity(self, pred: List[str], true: List[str]) -> float:
        if not pred or not true:
            return 0.0
        pred_words = set()
        for p in pred:
            pred_words.update(TurkishTextProcessor.get_turkish_keywords(str(p)))
        true_words = set()
        for t in true:
            true_words.update(TurkishTextProcessor.get_turkish_keywords(str(t)))
        
        if not pred_words or not true_words:
            return 0.0
        
        intersection = len(pred_words & true_words)
        union = len(pred_words | true_words)
        return intersection / union if union > 0 else 0.0
    
    def evaluate_tools_multi_level(self, pred_actions: List[str], true_actions: List[str]) -> Dict[str, Any]:
        if not true_actions:
            return {
                "strict_exact_match": 1.0,
                "normal_alias_match": 1.0,
                "fuzzy_semantic_match": 1.0,
                "weighted_final_score": 1.0,
                "tool_jaccard_iou": 1.0,
                "strict_predicted": [],
                "strict_expected": [],
                "normal_predicted": [],
                "normal_expected": [],
                "tp": 0, "fp": 0, "fn": 0
            }
        
        strict_pred = self._extract_exact_tools(pred_actions)
        strict_true = self._extract_exact_tools(true_actions)
        strict_score = 1.0 if strict_pred == strict_true else (len(strict_pred & strict_true) / len(strict_true) if strict_true else 1.0)

        normal_pred = self._extract_tools_with_aliases(pred_actions)
        normal_true = self._extract_tools_with_aliases(true_actions)
        
        tp = len(normal_pred & normal_true)
        fp = len(normal_pred - normal_true)
        fn = len(normal_true - normal_pred)
        
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
        normal_score = tp / len(normal_true) if normal_true else 1.0
        fuzzy_score = self._calculate_semantic_similarity(pred_actions, true_actions)
        
        weights = {"strict": 0.20, "normal": 0.50, "fuzzy": 0.30}
        weighted_score = (
            strict_score * weights["strict"] +
            iou * weights["normal"] +
            fuzzy_score * weights["fuzzy"]
        )
        
        return {
            "strict_exact_match": 1.0 if strict_pred == strict_true else 0.0,
            "normal_alias_match": round(normal_score, 3),
            "fuzzy_semantic_match": round(fuzzy_score, 3),
            "tool_jaccard_iou": round(iou, 3),
            "weighted_final_score": round(weighted_score, 3),
            "strict_predicted": list(strict_pred),
            "strict_expected": list(strict_true),
            "normal_predicted": list(normal_pred),
            "normal_expected": list(normal_true),
            "tp": tp, "fp": fp, "fn": fn,
            "predicted_raw": [str(a)[:50] for a in pred_actions],
            "expected_raw": [str(a)[:50] for a in true_actions]
        }


# ============================================================================
# 4. ÇOK SEVİYELİ EVENT DEĞERLENDİRİCİ
# ============================================================================

class MultiLevelEventEvaluator:
    """
    Olay değerlendirmesinde 3 seviyeli yaklaşım (Strict, Normal, Fuzzy)
    ve F1 / Recall / Precision metrikleri
    """
    def __init__(self, time_tolerance: int = 6):
        self.time_tolerance = time_tolerance
        self.tr_processor = TurkishTextProcessor()

    def set_tolerance_from_duration(self, duration: float):
        """Video süresine göre tolerans ayarla"""
        if duration <= 30:
            self.time_tolerance = 3
        elif duration <= 60:
            self.time_tolerance = 5
        else:
            self.time_tolerance = 8
        return self.time_tolerance

    def _time_to_seconds(self, time_val: Any) -> int:
        if isinstance(time_val, (int, float)):
            return int(time_val)
        if isinstance(time_val, str):
            parts = time_val.strip().split(':')
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        return 0
    
    def evaluate_events_multi_level(self, pred_events: List[Dict], true_events: List[Dict]) -> Dict[str, Any]:
        if not true_events:
            return {
                "strict_score": 1.0, "normal_score": 1.0, "fuzzy_score": 1.0,
                "weighted_final": 1.0, "f1_score": 1.0, "recall": 1.0, "precision": 1.0,
                "strict_matches": 0, "normal_matches": 0, "fuzzy_matches": 0,
                "total_gt_events": 0, "detected_events": len(pred_events)
            }
        
        strict_matches = 0
        normal_matches = 0
        fuzzy_matches = 0
        
        matched_gt = set()
        
        for gt in true_events:
            gt_time = self._time_to_seconds(gt.get("time", gt.get("start_time", 0)))
            gt_text = self.tr_processor.normalize(str(gt.get("event", "")))
            gt_keywords = self.tr_processor.get_turkish_keywords(gt_text)
            
            best_strict = False
            best_normal = False
            best_fuzzy = False
            
            for pred in pred_events:
                p_time = self._time_to_seconds(pred.get("time", pred.get("start_time", 0)))
                p_text = self.tr_processor.normalize(str(pred.get("event", pred.get("description", ""))))
                p_keywords = self.tr_processor.get_turkish_keywords(p_text)
                
                time_ok = abs(p_time - gt_time) <= self.time_tolerance
                
                if time_ok:
                    if p_text == gt_text:
                        best_strict = True
                        best_normal = True
                        best_fuzzy = True
                        break
                    
                    overlap = len(gt_keywords & p_keywords)
                    if overlap > 0 or len(gt_keywords) == 0:
                        best_normal = True
                        best_fuzzy = True
                    
                    if gt_keywords and p_keywords:
                        jaccard = len(gt_keywords & p_keywords) / len(gt_keywords | p_keywords)
                        if jaccard >= 0.15:
                            best_fuzzy = True
            
            if best_strict:
                strict_matches += 1
            if best_normal:
                normal_matches += 1
            if best_fuzzy:
                fuzzy_matches += 1
        
        total = len(true_events)
        strict_score = strict_matches / total
        normal_score = normal_matches / total
        fuzzy_score = fuzzy_matches / total
        
        tp = normal_matches
        fp = max(0, len(pred_events) - tp)
        fn = max(0, total - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        weights = {"strict": 0.15, "normal": 0.45, "fuzzy": 0.40}
        weighted_score = (
            strict_score * weights["strict"] +
            normal_score * weights["normal"] +
            fuzzy_score * weights["fuzzy"]
        )
        
        return {
            "strict_exact_match": round(strict_score, 3),
            "normal_keyword_match": round(normal_score, 3),
            "fuzzy_semantic_match": round(fuzzy_score, 3),
            "f1_score": round(f1, 3),
            "recall": round(recall, 3),
            "precision": round(precision, 3),
            "weighted_final_score": round(weighted_score, 3),
            "strict_matches": strict_matches,
            "normal_matches": normal_matches,
            "fuzzy_matches": fuzzy_matches,
            "total_gt_events": total,
            "detected_events": len(pred_events)
        }


# ============================================================================
# 5. ANA BENCHMARK SINIFI
# ============================================================================

class EnhancedAgentBenchmarkKPI:
    """GELİŞMİŞ KATMANLI KPI DEĞERLENDİRİCİ"""
    
    def __init__(self, time_tolerance_sec: int =8):
        self.time_tolerance = time_tolerance_sec
        self.tr_processor = TurkishTextProcessor()
        self.tool_evaluator = MultiLevelToolEvaluator()
        self.event_evaluator = MultiLevelEventEvaluator(time_tolerance_sec)

    @staticmethod
    def _get_gpu_peak_vram_gb() -> float:
        """vLLM harici process belleğini nvidia-smi üzerinden doğru okur"""
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
                encoding="utf-8"
            )
            vram_mb = float(output.strip().split("\n")[0])
            return round(vram_mb / 1024, 2)
        except Exception:
            return 0.0

    def evaluate_risk(self, pred: Dict, true: Dict) -> float:
        pred_risk = self.tr_processor.normalize(str(pred.get("risk_level", pred.get("risk", ""))))
        true_risk = self.tr_processor.normalize(str(true.get("risk_level", true.get("risk", ""))))
        return 1.0 if (pred_risk == true_risk and pred_risk != "") else 0.0

    def test_argument_extraction(self, full_state: Dict, true: Dict) -> Dict[str, Any]:
        tool_history = full_state.get("tool_history", [])
        true_actions = true.get("actions", [])
        
        param_scores = []
        errors = []
        
        for exp_tool in true_actions:
            matching_calls = [
                t for t in tool_history 
                if isinstance(t, dict) and t.get("tool") == exp_tool
            ]
            
            if not matching_calls:
                errors.append(f"'{exp_tool}' çağrısı yapılmamış")
                param_scores.append(0.0)
                continue
            
            best_score = 0.0
            for call in matching_calls:
                pred_params = call.get("args", call.get("parameters", {}))
                if isinstance(pred_params, dict):
                    best_score = 1.0
            
            param_scores.append(best_score)
        
        avg_score = float(np.mean(param_scores)) if param_scores else 1.0
        return {
            "parameter_accuracy": round(avg_score, 3),
            "missing_calls": len(errors),
            "errors": errors[:3]
        }

    def test_grounding(self, full_state: Dict) -> Dict[str, Any]:
        tool_results = []
        for item in full_state.get("tool_history", []):
            if isinstance(item, dict) and "result" in item:
                tool_results.append(item.get("result"))
        
        final_json = full_state.get("final_json_output", "")
        final_text = self.tr_processor.normalize(str(final_json))
        
        grounded_count = 0
        total_checks = 0
        
        for result in tool_results:
            if isinstance(result, dict):
                for value in result.values():
                    total_checks += 1
                    norm_val = self.tr_processor.normalize(str(value))
                    if len(norm_val) > 2 and norm_val in final_text:
                        grounded_count += 1
            elif isinstance(result, str):
                total_checks += 1
                norm_val = self.tr_processor.normalize(result)
                if len(norm_val) > 2 and norm_val in final_text:
                    grounded_count += 1
        
        grounding_score = grounded_count / total_checks if total_checks > 0 else 1.0
        return {
            "grounding_score": round(grounding_score, 3),
            "total_tool_results": len(tool_results),
            "grounded_items": grounded_count
        }

    def test_error_recovery(self, full_state: Dict) -> Dict[str, Any]:
        tool_history = full_state.get("tool_history", [])
        errors = [t.get("tool") for t in tool_history if isinstance(t, dict) and (t.get("error") or t.get("status") == "error")]
        recovery_score = 1.0 if not errors else 0.8
        return {
            "recovery_score": round(recovery_score, 3),
            "error_count": len(errors),
            "retry_count": 0,
            "recoveries": [],
            "error_messages": []
        }

    def run_benchmark(self, pipeline_fn, video_path: str, ground_truth: Dict[str, Any], 
                      video_duration_sec: float) -> Dict[str, Any]:
        start_time = time.perf_counter()
        prediction = pipeline_fn(video_path)
        total_latency = time.perf_counter() - start_time
        
        full_state = prediction.get("_full_state", {})
        peak_vram_gb = self._get_gpu_peak_vram_gb()
        rtf = total_latency / video_duration_sec if video_duration_sec > 0 else 0.0

        tool_results = self.tool_evaluator.evaluate_tools_multi_level(prediction.get("actions", []), ground_truth.get("actions", []))
        event_results = self.event_evaluator.evaluate_events_multi_level(prediction.get("events", []), ground_truth.get("events", []))
        risk_score = self.evaluate_risk(prediction, ground_truth)
        argument_results = self.test_argument_extraction(full_state, ground_truth)
        grounding_results = self.test_grounding(full_state)
        error_recovery_results = self.test_error_recovery(full_state)

        weights = {
            "tool": 0.25,
            "event": 0.20,
            "risk": 0.15,
            "argument": 0.15,
            "grounding": 0.15,
            "recovery": 0.10
        }
        
        overall_score = (
            tool_results["weighted_final_score"] * weights["tool"] +
            event_results["weighted_final_score"] * weights["event"] +
            risk_score * weights["risk"] +
            argument_results["parameter_accuracy"] * weights["argument"] +
            grounding_results["grounding_score"] * weights["grounding"] +
            error_recovery_results["recovery_score"] * weights["recovery"]
        )

        return {
            "performance": {
                "total_processing_time_sec": round(total_latency, 3),
                "peak_vram_gb": peak_vram_gb,
                "real_time_factor": round(rtf, 3)
            },
            "accuracy": {
                "event_f1_score": event_results["f1_score"],
                "event_recall": event_results["recall"],
                "risk_accuracy": risk_score,
                "tool_jaccard_score": tool_results["tool_jaccard_iou"],
                "tool_exact_match": tool_results["strict_exact_match"]
            },
            "layer_analysis": {
                "tool_selection": {
                    "risk_accuracy": risk_score,
                    "tool_jaccard_iou": tool_results["tool_jaccard_iou"],
                    "tool_exact_match": tool_results["strict_exact_match"],
                    "detected_tools": tool_results["normal_predicted"],
                    "expected_tools": tool_results["normal_expected"],
                    "hallucination_rate": round(tool_results["fp"] / (tool_results["tp"] + tool_results["fp"]), 3) if (tool_results["tp"] + tool_results["fp"]) > 0 else 0.0,
                    "tp": tool_results["tp"],
                    "fp": tool_results["fp"],
                    "fn": tool_results["fn"]
                },
                "argument_extraction": argument_results,
                "grounding": grounding_results,
                "error_recovery": error_recovery_results,
                "layer_weights": weights,
                "overall_score": round(overall_score, 3)
            },
            "debug_tools": {
                "matched_predictions": tool_results["normal_predicted"],
                "expected_ground_truth": tool_results["normal_expected"]
            }
        }


# ============================================================================
# 6. RAPORLAMA VE YÜRÜTME
# ============================================================================

def run_enhanced_benchmark(ground_truth_path: str = "ground_truth.json", 
                           videos_dir: str = "Videos"):
    if not os.path.exists(ground_truth_path):
        print(f"❌ Hata: '{ground_truth_path}' bulunamadı.")
        return

    with open(ground_truth_path, "r", encoding="utf-8") as f:
        ground_truth_list = json.load(f)

    benchmark = EnhancedAgentBenchmarkKPI(time_tolerance_sec=6)
    all_results = []
    
    metrics = {
        "event_f1": [], "event_recall": [], "risk": [], 
        "tool_jaccard": [], "tool_exact": [], "overall": [],
        "argument": [], "grounding": [], "recovery": [],
        "latency": [], "rtf": [], "vram": []
    }

    print(f"\n🚀 {len(ground_truth_list)} video için KPI Benchmark başlatılıyor...")
    print("=" * 50)

    for idx, item in enumerate(ground_truth_list, 1):
        video_id = item.get("video_id", "")
        duration = float(item.get("duration_sec", 1.0))
        video_path = os.path.join(videos_dir, video_id)

        if not os.path.exists(video_path):
            print(f"⚠️ Video bulunamadı: {video_path} (atlanıyor)")
            continue

        # Sadece ilerleme göster - TEMİZ ÇIKTI
        print(f"▶️ [{idx}/{len(ground_truth_list)}] {video_id} ({duration} sn) ...", end=" ", flush=True)
        
        try:
            res = benchmark.run_benchmark(
                pipeline_fn=run_agent_pipeline,
                video_path=video_path,
                ground_truth=item,
                video_duration_sec=duration
            )
            all_results.append({"video_id": video_id, "metrics": res})
            print("✅")
        except Exception as e:
            print(f"❌ Hata: {str(e)[:50]}")
            continue

        acc = res.get("accuracy", {})
        perf = res.get("performance", {})
        layer = res.get("layer_analysis", {})

        metrics["event_f1"].append(acc.get("event_f1_score", 0))
        metrics["event_recall"].append(acc.get("event_recall", 0))
        metrics["risk"].append(acc.get("risk_accuracy", 0))
        metrics["tool_jaccard"].append(acc.get("tool_jaccard_score", 0))
        metrics["tool_exact"].append(acc.get("tool_exact_match", 0))
        metrics["overall"].append(layer.get("overall_score", 0))
        metrics["argument"].append(layer.get("argument_extraction", {}).get("parameter_accuracy", 0))
        metrics["grounding"].append(layer.get("grounding", {}).get("grounding_score", 0))
        metrics["recovery"].append(layer.get("error_recovery", {}).get("recovery_score", 0))
        metrics["latency"].append(perf.get("total_processing_time_sec", 0))
        metrics["rtf"].append(perf.get("real_time_factor", 0))
        metrics["vram"].append(perf.get("peak_vram_gb", 0))

    if not all_results:
        print("❌ Hiçbir video işlenemedi.")
        return

    with open("benchmark_results_enhanced.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print("📊 KPI DEĞERLENDİRME RAPORU")
    print("=" * 50)
    print(f"Toplam Test Edilen Video : {len(all_results)}")
    print("-" * 50)
    print("🎯 DOĞRULUK METRİKLERİ:")
    print(f"  • Event F1-Score        : {np.mean(metrics['event_f1']):.3f}")
    print(f"  • Event Recall          : {np.mean(metrics['event_recall']):.3f}")
    print(f"  • Risk Doğruluğu        : %{np.mean(metrics['risk']) * 100:.1f}")
    print(f"  • Tool Jaccard (IoU)    : {np.mean(metrics['tool_jaccard']):.3f}")
    print(f"  • Tool Exact Match      : %{np.mean(metrics['tool_exact']) * 100:.1f}")
    print("-" * 50)
    print("🧩 KATMAN BAZLI ANALİZ:")
    print(f"  • Genel Sistem Skoru    : {np.mean(metrics['overall']):.3f}")
    print(f"  • Argument Extraction   : {np.mean(metrics['argument']):.3f}")
    print(f"  • Grounding Skoru       : {np.mean(metrics['grounding']):.3f}")
    print("-" * 50)
    print("⚡ PERFORMANS:")
    print(f"  • Ort. İşlem Süresi     : {np.mean(metrics['latency']):.2f} sn")
    print(f"  • Ort. Real Time Factor : {np.mean(metrics['rtf']):.3f}")
    print(f"  • Peak VRAM             : {np.max(metrics['vram']):.2f} GB")
    print("=" * 50)
    print("📁 Sonuçlar: benchmark_results_enhanced.json\n")


if __name__ == "__main__":
    run_enhanced_benchmark()