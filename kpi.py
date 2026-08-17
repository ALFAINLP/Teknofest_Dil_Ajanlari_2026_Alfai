import json
import os
import re
import time
import numpy as np
import torch
from typing import List, Dict, Any, Set

# LangGraph ajan bileşenlerini dahil ediyoruz
from agent.graph import app
from agent.memory_manager import MemoryManager


def run_agent_pipeline(video_path: str) -> dict:
    """
    main.py içerisindeki LangGraph akışını her video için bağımsız çalıştırır
    ve KPI hesaplaması için standart bir dict formatına dönüştürür.
    """
    memory = MemoryManager.load_memory()

    initial_state = {
        "video_path": video_path,
        "current_time_segments": [],
        "detected_events": [],
        "risk_assessments": [],
        "actions_recommended": [],
        "final_json_output": "",
        "next_step": "",
        "event_history": memory.get("event_history", []),
        "risk_history": memory.get("risk_history", []),
        "tool_history": memory.get("tool_history", []),
        "messages": []
    }

    # LangGraph ReAct döngüsünü tetikle
    final_output = app.invoke(initial_state)

    # 1. Çağrılan Tool/Aksiyon isimlerini yakala
    collected_actions = []
    for msg in final_output.get("messages", []):
        tool_calls = getattr(msg, "tool_calls", [])
        if tool_calls:
            for tc in tool_calls:
                if isinstance(tc, dict) and "name" in tc:
                    collected_actions.append(tc["name"])

    # 2. Nihai JSON çıktısını çözümle
    final_json = final_output.get("final_json_output", "")
    parsed_report = {}

    if isinstance(final_json, str) and final_json.strip():
        try:
            # Markdown code-block varsa temizle
            cleaned_json = re.sub(r"^```json\s*|\s*```$", "", final_json.strip(), flags=re.MULTILINE)
            parsed_report = json.loads(cleaned_json)
        except Exception:
            parsed_report = {}
    elif isinstance(final_json, dict):
        parsed_report = final_json

    # Model aksiyon listesini json çıktısında da vermiş olabilir, birleştir
    if "actions" in parsed_report and isinstance(parsed_report["actions"], list):
        collected_actions.extend(parsed_report["actions"])

    return {
        "risk": parsed_report.get("risk", parsed_report.get("risk_level", "Bilinmiyor")),
        "events": parsed_report.get("events", parsed_report.get("detected_events", [])),
        "actions": list(set(collected_actions))
    }


class AgentBenchmarkKPI:
    def __init__(self, time_tolerance_sec: int = 3):
        self.time_tolerance = time_tolerance_sec

    def _time_to_seconds(self, time_val: Any) -> int:
        """'00:15' formatını veya int/float saniye değerini saniyeye çevirir."""
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
        """Tool isimlerini regex ile normalize eder."""
        extracted = set()
        for item in action_list:
            item = str(item).strip().lower()
            quotes_match = re.findall(r"['\"]([a-zA-Z0-9_]+)['\"]", item)
            if quotes_match:
                extracted.update(quotes_match)
            else:
                cleaned = re.sub(r"[^a-zA-Z0-9_]", "", item)
                if cleaned:
                    extracted.add(cleaned)
        return extracted

    def evaluate_events(self, pred_events: List[Dict], true_events: List[Dict]) -> Dict[str, float]:
        """Zaman damgalı olay tespiti için F1, Precision ve Recall hesaplar."""
        tp, fp = 0, 0
        matched_gt = set()

        for pred in pred_events:
            p_time = self._time_to_seconds(pred.get("time", pred.get("start_time", 0)))
            p_desc = str(pred.get("event", pred.get("description", ""))).lower()
            match_found = False

            for i, gt in enumerate(true_events):
                if i in matched_gt:
                    continue
                g_time = self._time_to_seconds(gt.get("time", gt.get("start_time", 0)))
                g_desc = str(gt.get("event", "")).lower()

                if abs(p_time - g_time) <= self.time_tolerance:
                    if any(w in g_desc for w in p_desc.split() if len(w) > 2):
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

    def evaluate_actions_and_risk(self, pred: Dict, true: Dict) -> Dict[str, Any]:
        """Risk ve Tool çağrıları üzerinden Jaccard ve Exact Match hesaplar."""
        pred_risk = str(pred.get("risk", "")).strip().lower()
        true_risk = str(true.get("risk", "")).strip().lower()
        risk_score = 1.0 if (pred_risk == true_risk and pred_risk != "") else 0.0

        pred_tools = self._extract_tool_names(pred.get("actions", []))
        true_tools = self._extract_tool_names(true.get("actions", []))

        intersection = len(pred_tools.intersection(true_tools))
        union = len(pred_tools.union(true_tools))

        tool_jaccard_iou = intersection / union if union > 0 else 0.0
        tool_exact_match = 1.0 if pred_tools == true_tools and len(true_tools) > 0 else 0.0

        return {
            "risk_accuracy": risk_score,
            "tool_jaccard_iou": round(tool_jaccard_iou, 3),
            "tool_exact_match": tool_exact_match,
            "detected_tools": list(pred_tools),
            "expected_tools": list(true_tools)
        }

    def run_benchmark(self, pipeline_fn, video_path: str, ground_truth: Dict[str, Any], video_duration_sec: float) -> Dict[str, Any]:
        """Tek video için donanım/zaman ölçümü yapar ve metrikleri hesaplar."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        
        # Ajanı çalıştır
        prediction = pipeline_fn(video_path)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_latency = time.perf_counter() - start_time

        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
        rtf = total_latency / video_duration_sec if video_duration_sec > 0 else 0.0

        event_kpis = self.evaluate_events(prediction.get("events", []), ground_truth.get("events", []))
        decision_kpis = self.evaluate_actions_and_risk(prediction, ground_truth)

        return {
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
            }
        }


def run_full_benchmark(ground_truth_path: str = "ground_truth.json", videos_dir: str = "Videos"):
    if not os.path.exists(ground_truth_path):
        print(f"❌ Hata: '{ground_truth_path}' dosyası bulunamadı.")
        return

    with open(ground_truth_path, "r", encoding="utf-8") as f:
        ground_truth_list = json.load(f)

    benchmark = AgentBenchmarkKPI(time_tolerance_sec=3)
    all_results = []
    
    f1_scores, recalls, risk_accs, tool_jaccards, tool_exacts, rtf_scores, vram_usages, latencies = [], [], [], [], [], [], [], []

    print(f"\n🚀 {len(ground_truth_list)} video için KPI Benchmark başlatılıyor...\n" + "="*70)

    for item in ground_truth_list:
        video_id = item.get("video_id", "")
        duration = float(item.get("duration_sec", 1.0))
        video_path = os.path.join(videos_dir, video_id)

        if not os.path.exists(video_path):
            print(f"⚠️ Video bulunamadı: {video_path} (atlanıyor)")
            continue

        print(f"▶️ Test ediliyor: {video_id} ({duration} sn)")

        res = benchmark.run_benchmark(
            pipeline_fn=run_agent_pipeline,
            video_path=video_path,
            ground_truth=item,
            video_duration_sec=duration
        )

        all_results.append({"video_id": video_id, "metrics": res})

        f1_scores.append(res["accuracy"]["event_f1_score"])
        recalls.append(res["accuracy"]["event_recall"])
        risk_accs.append(res["accuracy"]["risk_accuracy"])
        tool_jaccards.append(res["accuracy"]["tool_jaccard_score"])
        tool_exacts.append(res["accuracy"]["tool_exact_match"])
        rtf_scores.append(res["performance"]["real_time_factor"])
        vram_usages.append(res["performance"]["peak_vram_gb"])
        latencies.append(res["performance"]["total_processing_time_sec"])

    if not all_results:
        print("❌ Hiçbir video işlenemedi. 'Videos/' klasöründeki video isimlerini kontrol edin.")
        return

    print("\n" + "="*70)
    print("📊 TEKNOFEST KPI DEĞERLENDİRME RAPORU")
    print("="*70)
    print(f"Toplam Test Edilen Video : {len(all_results)}")
    print("-" * 70)
    print("🎯 DOĞRULUK METRİKLERİ:")
    print(f"  • Ortalama Event F1-Score : {np.mean(f1_scores):.3f}")
    print(f"  • Ortalama Event Recall   : {np.mean(recalls):.3f}")
    print(f"  • Risk Doğruluğu (Acc)    : %{np.mean(risk_accs) * 100:.1f}")
    print(f"  • Tool Jaccard (IoU)      : {np.mean(tool_jaccards):.3f}")
    print(f"  • Tool Exact Match        : %{np.mean(tool_exacts) * 100:.1f}")
    print("-" * 70)
    print("⚡ PERFORMANS METRİKLERİ:")
    print(f"  • Ort. İşlem Süresi (sn)  : {np.mean(latencies):.2f} sn")
    print(f"  • Ort. Real Time Factor   : {np.mean(rtf_scores):.3f} (RTF < 1.0 = Gerçek zamandan hızlı)")
    print(f"  • Peak VRAM Kullanımı     : {np.max(vram_usages):.2f} GB")
    print("="*70)

    with open("benchmark_results.json", "w", encoding="utf-8") as f_out:
        json.dump(all_results, f_out, indent=2, ensure_ascii=False)
    print("📁 Detaylı loglar 'benchmark_results.json' dosyasına kaydedildi.\n")


if __name__ == "__main__":
    run_full_benchmark(ground_truth_path="ground_truth.json", videos_dir="Videos")
