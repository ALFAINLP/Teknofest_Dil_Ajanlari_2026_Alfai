# kpi_enhanced.py - SİZİN KODUNUZUN GELİŞMİŞ HALİ

import json
import os
import re
import time
import numpy as np
import torch
from typing import List, Dict, Any, Set, Optional, Tuple
from collections import defaultdict

from agent.graph import app  # Sizin mevcut import'unuz

# ============================================================================
# 1. SİZİN MEVCUT run_agent_pipeline FONKSİYONUNUZ (AYNEN KORUNDU)
# ============================================================================

def run_agent_pipeline(video_path: str) -> dict:
    """
    Her video için sıfırdan temiz bir state ile LangGraph pipeline'ını çalıştırır.
    Hafıza kirliliğini önlemek için tool_history sıfır başlatılır.
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

    final_output = app.invoke(initial_state)  # Sizin mevcut çağrınız

    # 1. JSON çıktısını çözümle - SİZİN KODUNUZ
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

    # 2. Risk ve Olaylar - SİZİN KODUNUZ
    detected_risk = parsed_report.get("risk", final_output.get("risk_assessments", ["Düşük"])[0])
    detected_events = parsed_report.get("events", final_output.get("detected_events", []))
    
    # 3. Actions / Tools
    # actions_recommended, graph.py'de risk seviyesine göre filtrelenmiş hali içeriyor.
    # tool_history'den tekrar eklemiyoruz — Düşük risk filtrelemesini bozmamak için.
    executed_tools = set(final_output.get("actions_recommended", []))

    return {
        "risk_level": detected_risk,
        "events": detected_events,
        "actions": list(executed_tools),
        # Ek olarak: GELİŞMİŞ TESTLER İÇİN TAM STATE'İ DE DÖNDÜR
        "_full_state": final_output  
    }



# ============================================================================
# 2. GELİŞMİŞ KATMANLI TEST SINIFI (MEVCUT KODUN ÜZERİNE EKLENDİ)
# ============================================================================

class TurkishTextProcessor:
    """Türkçe metin işleme için yardımcı sınıf"""
    
    @staticmethod
    def normalize(text: str) -> str:
        """Türkçe karakter normalizasyonu"""
        replacements = {
            'ı': 'i', 'ğ': 'g', 'ü': 'u', 'ş': 's', 'ö': 'o', 'ç': 'c',
            'I': 'I', 'Ğ': 'G', 'Ü': 'U', 'Ş': 'S', 'Ö': 'O', 'Ç': 'C'
        }
        for tr_char, en_char in replacements.items():
            text = text.replace(tr_char, en_char)
        return text.lower()
    
    @staticmethod
    def get_turkish_keywords(text: str, min_len: int = 2) -> Set[str]:
        """Türkçe metinden anlamlı kelimeler çıkar"""
        if not text:
            return set()
        
        # Türkçe stop words
        tr_stop_words = {
            've', 'ile', 'için', 'bir', 'bu', 'şu', 'o', 'da', 'de', 'ki',
            'mi', 'mu', 'mı', 'mü', 'ne', 'nasıl', 'neden', 'niye', 'çünkü',
            'ama', 'fakat', 'ancak', 'lakin', 'yani', 'demek', 'şöyle', 'böyle',
            'öyle', 'şimdi', 'sonra', 'önce', 'şimdi', 'sonra', 'önce'
        }
        
        # Türkçe karakterleri normalize et
        normalized = TurkishTextProcessor.normalize(text)
        words = set(re.findall(r'[a-zğüşıöç]{2,}', normalized.lower()))
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
        
        return len(words1.intersection(words2)) / len(words1.union(words2))


class EnhancedAgentBenchmarkKPI:
    """
    GELİŞMİŞ KPI TEST - Sizin mevcut kodunuzu kullanır, katmanlı test ekler
    """
    
    def __init__(self, time_tolerance_sec: int = 6):
        self.time_tolerance = time_tolerance_sec
        self.tr_processor = TurkishTextProcessor()

    def _time_to_seconds(self, time_val: Any) -> int:
        """Sizin mevcut _time_to_seconds fonksiyonunuz"""
        if isinstance(time_val, (int, float)):
            return int(time_val)
        if isinstance(time_val, str):
            parts = time_val.strip().split(':')
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        return 0

    def _extract_tool_names(self, action_list: List[str]) -> Set[str]:
        """Sizin mevcut _extract_tool_names fonksiyonunuz"""
        valid_tools = {
            "trigger_operator_alert", "get_personnel_on_duty",
            "assess_environmental_hazards", "query_previous_incidents",
            "export_to_incident_management_system", "detect_motion_density",
            "log_kpi_metrics"  # Düşük risk senaryolarında GT tarafından bekleniyor
        }
        extracted = set()
        for item in action_list:
            item_str = str(item).lower()
            for vt in valid_tools:
                if vt in item_str:
                    extracted.add(vt)
        return extracted


    # ===== SİZİN MEVCUT evaluate_events FONKSİYONUNUZ (GELİŞTİRİLDİ) =====
    def evaluate_events(self, pred_events: List[Dict], true_events: List[Dict]) -> Dict[str, float]:
        """Sizin mevcut fonksiyonunuz - Türkçe desteği ile geliştirildi"""
        tp, fp = 0, 0
        matched_gt = set()

        for pred in pred_events:
            p_time = self._time_to_seconds(pred.get("time", pred.get("start_time", 0)))
            p_desc = str(pred.get("event", pred.get("description", "")))
            match_found = False

            for i, gt in enumerate(true_events):
                if i in matched_gt:
                    continue
                g_time = self._time_to_seconds(gt.get("time", gt.get("start_time", 0)))
                g_desc = str(gt.get("event", ""))

                if abs(p_time - g_time) <= self.time_tolerance:
                    # TÜRKÇE JACCARD BENZERLİĞİ KULLAN
                    similarity = self.tr_processor.turkish_jaccard(p_desc, g_desc)
                    if similarity >= 0.3:  # 0.3 eşik değeri
                        tp += 1
                        matched_gt.add(i)
                        match_found = True
                        break
            if not match_found:
                fp += 1

        fn = len(true_events) - len(matched_gt)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"precision": precision, "recall": recall, "f1_score": f1}

    # ===== SİZİN MEVCUT evaluate_actions_and_risk FONKSİYONUNUZ =====
    def evaluate_actions_and_risk(self, pred: Dict, true: Dict) -> Dict[str, Any]:
        """Sizin mevcut fonksiyonunuz - aynen korundu"""
        pred_risk = str(pred.get("risk_level", pred.get("risk", ""))).strip().lower()
        true_risk = str(true.get("risk_level", true.get("risk", ""))).strip().lower()
        
        # Türkçe risk seviyeleri için normalizasyon
        risk_score = 1.0 if (pred_risk == true_risk and pred_risk != "") else 0.0

        pred_tools = self._extract_tool_names(pred.get("actions", []))
        true_tools = self._extract_tool_names(true.get("actions", []))

        intersection = len(pred_tools.intersection(true_tools))
        union = len(pred_tools.union(true_tools))

        if len(pred_tools) == 0 and len(true_tools) == 0:
            tool_jaccard_iou = 1.0
            tool_exact_match = 1.0
        else:
            tool_jaccard_iou = intersection / union if union > 0 else 0.0
            tool_exact_match = 1.0 if pred_tools == true_tools else 0.0

        return {
            "risk_accuracy": risk_score,
            "tool_jaccard_iou": round(tool_jaccard_iou, 3),
            "tool_exact_match": tool_exact_match,
            "detected_tools": list(pred_tools),
            "expected_tools": list(true_tools)
        }

    # ===== YENİ: KATMAN 1 - TOOL SELECTION TESTİ =====
    def test_tool_selection_layer(self, pred: Dict, true: Dict) -> Dict[str, Any]:
        """
        YENİ KATMAN: Ajan doğru araçları seçebiliyor mu?
        Mevcut evaluate_actions_and_risk'i kullanır, üzerine ek metrikler ekler
        """
        base_metrics = self.evaluate_actions_and_risk(pred, true)
        
        # Tool çağrılarını analiz et
        pred_tools = set(base_metrics["detected_tools"])
        true_tools = set(base_metrics["expected_tools"])
        
        # Hallüsinasyon oranı (gereksiz tool çağrıları)
        tp = len(pred_tools.intersection(true_tools))
        fp = len(pred_tools - true_tools)
        hallucination_rate = fp / len(pred_tools) if pred_tools else 0.0
        
        return {
            **base_metrics,
            "hallucination_rate": round(hallucination_rate, 3),
            "tp": tp,
            "fp": fp,
            "fn": len(true_tools - pred_tools)
        }

    # ===== YENİ: KATMAN 2 - ARGUMENT EXTRACTION TESTİ =====
    def test_argument_extraction_layer(self, full_state: Dict, true: Dict) -> Dict[str, Any]:
        """
        YENİ KATMAN: Ajan tool parametrelerini doğru çıkarıyor mu?
        """
        tool_history = full_state.get("tool_history", [])
        expected_tools = true.get("actions", [])
        
        param_scores = []
        errors = []
        
        for exp_tool in expected_tools:
            # Bu tool için yapılan çağrıları bul
            matching_calls = [
                t for t in tool_history 
                if isinstance(t, dict) and t.get("tool") == exp_tool
            ]
            
            if not matching_calls:
                errors.append(f"'{exp_tool}' çağrısı yapılmamış")
                param_scores.append(0.0)
                continue
            
            # Parametre eşleşme skoru
            best_score = 0.0
            for call in matching_calls:
                pred_params = call.get("parameters", {})
                # true params kontrolü - ground truth'da parametre varsa
                if isinstance(exp_tool, dict):
                    gt_params = exp_tool.get("parameters", {})
                    if gt_params:
                        matched = sum(1 for k in gt_params if k in pred_params)
                        score = matched / len(gt_params)
                        best_score = max(best_score, score)
                    else:
                        best_score = 1.0
                else:
                    best_score = 1.0  # Parametre beklenmiyorsa tam puan
            
            param_scores.append(best_score)
        
        avg_score = np.mean(param_scores) if param_scores else 1.0
        
        return {
            "parameter_accuracy": round(avg_score, 3),
            "missing_calls": len([e for e in errors if "çağrısı yapılmamış" in e]),
            "errors": errors[:3]
        }

    # ===== YENİ: KATMAN 3 - GROUNDING (TEMELLENDİRME) TESTİ =====
    def test_grounding_layer(self, full_state: Dict) -> Dict[str, Any]:
        """
        YENİ KATMAN: Ajan tool sonuçlarını nihai çıktısında kullanıyor mu?
        """
        tool_results = []
        for item in full_state.get("tool_history", []):
            if isinstance(item, dict) and "result" in item:
                tool_results.append(item.get("result"))
        
        # Final JSON çıktısı
        final_json = full_state.get("final_json_output", "")
        if isinstance(final_json, str) and final_json.strip():
            try:
                cleaned = re.sub(r"^```json\s*|\s*```$", "", final_json.strip(), flags=re.MULTILINE)
                final_output = json.loads(cleaned)
            except:
                final_output = {}
        else:
            final_output = {}
        
        final_text = json.dumps(final_output, ensure_ascii=False).lower()
        
        grounded_count = 0
        total_checks = 0
        
        for result in tool_results:
            if isinstance(result, dict):
                for value in result.values():
                    total_checks += 1
                    if isinstance(value, str) and value.lower() in final_text:
                        grounded_count += 1
                    elif isinstance(value, (int, float)) and str(value) in final_text:
                        grounded_count += 1
            elif isinstance(result, str) and result.lower() in final_text:
                grounded_count += 1
                total_checks += 1
        
        grounding_score = grounded_count / total_checks if total_checks > 0 else 1.0
        
        return {
            "grounding_score": round(grounding_score, 3),
            "total_tool_results": len(tool_results),
            "grounded_items": grounded_count
        }

    # ===== YENİ: KATMAN 4 - ERROR RECOVERY TESTİ =====
    def test_error_recovery_layer(self, full_state: Dict) -> Dict[str, Any]:
        """
        YENİ KATMAN: Ajan hatalardan kurtulabiliyor mu?
        """
        tool_history = full_state.get("tool_history", [])
        messages = full_state.get("messages", [])
        
        errors = []
        retries = 0
        recoveries = []
        
        for i, item in enumerate(tool_history):
            if not isinstance(item, dict):
                continue
            
            # Hata durumu
            if item.get("error") or item.get("status") == "error":
                errors.append(item.get("tool", "unknown"))
                
                # Retry kontrolü
                if i + 1 < len(tool_history):
                    next_tool = tool_history[i + 1].get("tool") if isinstance(tool_history[i + 1], dict) else None
                    if next_tool == item.get("tool"):
                        retries += 1
                    elif next_tool:
                        recoveries.append(f"{item.get('tool')} -> {next_tool}")
        
        # Hata mesajları
        error_msgs = [m for m in messages if isinstance(m, str) and "hata" in m.lower()]
        
        recovery_score = 1.0
        if errors:
            recovery_score = retries / len(errors) if errors else 0.0
            if recovery_score > 0.5:
                recovery_score = 1.0
        
        return {
            "recovery_score": round(recovery_score, 3),
            "error_count": len(errors),
            "retry_count": retries,
            "recoveries": recoveries[:3],
            "error_messages": error_msgs[:3]
        }

    # ===== ANA BENCHMARK FONKSİYONU (SİZİN KODUNUZ + YENİ KATMANLAR) =====
    def run_benchmark(self, pipeline_fn, video_path: str, ground_truth: Dict[str, Any], 
                      video_duration_sec: float) -> Dict[str, Any]:
        """
        SİZİN MEVCUT run_benchmark FONKSİYONUNUZ + KATMANLI TESTLER
        """
        # Hafıza temizliği - SİZİN KODUNUZ
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        prediction = pipeline_fn(video_path)  # SİZİN MEVCUT ÇAĞRI
        
        # FULL STATE'i al (ek katman testleri için)
        full_state = prediction.get("_full_state", {})
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_latency = time.perf_counter() - start_time

        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
        rtf = total_latency / video_duration_sec if video_duration_sec > 0 else 0.0

        # ===== SİZİN MEVCUT METRİKLER =====
        event_kpis = self.evaluate_events(prediction.get("events", []), ground_truth.get("events", []))
        decision_kpis = self.evaluate_actions_and_risk(prediction, ground_truth)

        # ===== YENİ KATMANLI METRİKLER =====
        tool_selection_layer = self.test_tool_selection_layer(prediction, ground_truth)
        argument_layer = self.test_argument_extraction_layer(full_state, ground_truth)
        grounding_layer = self.test_grounding_layer(full_state)
        error_recovery_layer = self.test_error_recovery_layer(full_state)

        # ===== GENEL SKOR (Ağırlıklı ortalama) =====
        layer_scores = {
            "tool_selection": tool_selection_layer["tool_jaccard_iou"],
            "argument_extraction": argument_layer["parameter_accuracy"],
            "grounding": grounding_layer["grounding_score"],
            "error_recovery": error_recovery_layer["recovery_score"],
            "event_f1": event_kpis["f1_score"],
            "risk_accuracy": decision_kpis["risk_accuracy"]
        }
        
        weights = {
            "tool_selection": 0.20,
            "argument_extraction": 0.15,
            "grounding": 0.15,
            "error_recovery": 0.15,
            "event_f1": 0.20,
            "risk_accuracy": 0.15
        }
        
        overall_score = sum(
            layer_scores[key] * weights[key] 
            for key in weights.keys() if key in layer_scores
        )

        return {
            # SİZİN MEVCUT ÇIKTI YAPISI (TAMAMEN KORUNDU)
            "performance": {
                "total_processing_time_sec": round(total_latency, 3),
                "peak_vram_gb": round(peak_vram_gb, 2),
                "real_time_factor": round(rtf, 3)
            },
            "accuracy": {
                "event_f1_score": round(event_kpis["f1_score"], 3),
                "event_recall": round(event_kpis["recall"], 3),
                "risk_accuracy": decision_kpis["risk_accuracy"],
                "tool_jaccard_score": decision_kpis["tool_jaccard_iou"],
                "tool_exact_match": decision_kpis["tool_exact_match"]
            },
            "debug_tools": {
                "matched_predictions": decision_kpis["detected_tools"],
                "expected_ground_truth": decision_kpis["expected_tools"]
            },
            # YENİ KATMANLI TEST SONUÇLARI
            "layer_analysis": {
                "tool_selection": tool_selection_layer,
                "argument_extraction": argument_layer,
                "grounding": grounding_layer,
                "error_recovery": error_recovery_layer,
                "layer_weights": weights,
                "overall_score": round(overall_score, 3)
            }
        }


# ============================================================================
# 3. ANA BENCHMARK FONKSİYONU (SİZİN MEVCUT KODUNUZ + GELİŞTİRMELER)
# ============================================================================

def run_enhanced_benchmark(ground_truth_path: str = "ground_truth.json", 
                           videos_dir: str = "Videos"):
    """
    SİZİN MEVCUT run_full_benchmark FONKSİYONU + GELİŞTİRMELER
    """
    if not os.path.exists(ground_truth_path):
        print(f"❌ Hata: '{ground_truth_path}' bulunamadı.")
        return

    with open(ground_truth_path, "r", encoding="utf-8") as f:
        ground_truth_list = json.load(f)

    benchmark = EnhancedAgentBenchmarkKPI(time_tolerance_sec=6)
    all_results = []
    
    # Sizin mevcut metrik listeleriniz
    f1_scores, recalls, risk_accs, tool_jaccards, tool_exacts = [], [], [], [], []
    rtf_scores, vram_usages, latencies = [], [], []
    
    # YENİ: Katman metrikleri için listeler
    overall_scores, tool_selection_scores, argument_scores = [], [], []
    grounding_scores, recovery_scores = [], []

    print(f"\n🚀 {len(ground_truth_list)} video için GELİŞMİŞ KPI Benchmark başlatılıyor...")
    print("="*80)

    for item in ground_truth_list:
        video_id = item.get("video_id", "")
        duration = float(item.get("duration_sec", 1.0))
        video_path = os.path.join(videos_dir, video_id)

        if not os.path.exists(video_path):
            print(f"⚠️ Video bulunamadı: {video_path} (atlanıyor)")
            continue

        print(f"▶️ Test ediliyor: {video_id} ({duration} sn)")

        res = benchmark.run_benchmark(
            pipeline_fn=run_agent_pipeline,  # Sizin mevcut pipeline'ınız
            video_path=video_path,
            ground_truth=item,
            video_duration_sec=duration
        )

        all_results.append({"video_id": video_id, "metrics": res})

        # SİZİN MEVCUT METRİKLER
        f1_scores.append(res["accuracy"]["event_f1_score"])
        recalls.append(res["accuracy"]["event_recall"])
        risk_accs.append(res["accuracy"]["risk_accuracy"])
        tool_jaccards.append(res["accuracy"]["tool_jaccard_score"])
        tool_exacts.append(res["accuracy"]["tool_exact_match"])
        rtf_scores.append(res["performance"]["real_time_factor"])
        vram_usages.append(res["performance"]["peak_vram_gb"])
        latencies.append(res["performance"]["total_processing_time_sec"])
        
        # YENİ KATMAN METRİKLERİ
        layer = res.get("layer_analysis", {})
        overall_scores.append(layer.get("overall_score", 0))
        tool_selection_scores.append(layer.get("tool_selection", {}).get("tool_jaccard_iou", 0))
        argument_scores.append(layer.get("argument_extraction", {}).get("parameter_accuracy", 0))
        grounding_scores.append(layer.get("grounding", {}).get("grounding_score", 0))
        recovery_scores.append(layer.get("error_recovery", {}).get("recovery_score", 0))

    if not all_results:
        print("❌ Hiçbir video işlenemedi.")
        return

    print("\n" + "="*80)
    print("📊 TEKNOFEST KPI DEĞERLENDİRME RAPORU (GELİŞMİŞ KATMANLI TEST)")
    print("="*80)
    print(f"Toplam Test Edilen Video : {len(all_results)}")
    print("-"*80)
    
    print("🎯 DOĞRULUK METRİKLERİ (SİZİN MEVCUT METRİKLERİNİZ):")
    print(f"  • Ortalama Event F1-Score : {np.mean(f1_scores):.3f}")
    print(f"  • Ortalama Event Recall   : {np.mean(recalls):.3f}")
    print(f"  • Risk Doğruluğu (Acc)    : %{np.mean(risk_accs) * 100:.1f}")
    print(f"  • Tool Jaccard (IoU)      : {np.mean(tool_jaccards):.3f}")
    print(f"  • Tool Exact Match        : %{np.mean(tool_exacts) * 100:.1f}")
    print("-"*80)
    
    print("🧩 KATMAN BAZLI ANALİZ (YENİ):")
    print(f"  • Genel Sistem Skoru      : {np.mean(overall_scores):.3f}")
    print(f"  • Tool Selection Skoru    : {np.mean(tool_selection_scores):.3f}")
    print(f"  • Argument Extraction     : {np.mean(argument_scores):.3f}")
    print(f"  • Grounding Skoru         : {np.mean(grounding_scores):.3f}")
    print(f"  • Error Recovery Skoru    : {np.mean(recovery_scores):.3f}")
    print("-"*80)
    
    print("⚡ PERFORMANS METRİKLERİ:")
    print(f"  • Ort. İşlem Süresi (sn)  : {np.mean(latencies):.2f} sn")
    print(f"  • Ort. Real Time Factor   : {np.mean(rtf_scores):.3f} (RTF < 1.0 = Gerçek zamandan hızlı)")
    print(f"  • Peak VRAM Kullanımı     : {np.max(vram_usages):.2f} GB")
    print("="*80)

    # Detaylı sonuçları kaydet
    with open("benchmark_results_enhanced.json", "w", encoding="utf-8") as f_out:
        json.dump(all_results, f_out, indent=2, ensure_ascii=False)
    print("📁 Detaylı loglar 'benchmark_results_enhanced.json' dosyasına kaydedildi.\n")


if __name__ == "__main__":
    run_enhanced_benchmark(ground_truth_path="ground_truth.json", videos_dir="Videos")