"""
test_single.py
--------------
Tek bir video için kpi_enhanced.py altyapısını kullanarak TAM KPI ölçümü yapar.
Olay tespiti (F1), Özet Kalitesi, Araç Eşleşmesi (Exact / IoU), Risk ve Performans
metriklerini TEKNOFEST şartnamesine tam uyumlu olarak terminale basar.

Kullanım:
    python test_single.py <video_path> [ground_truth_json_path] [ground_truth_video_id]

Örnek:
    python test_single.py Videos/test_video_008.mp4 ground_truth.json test_video_008.mp4
"""

import sys
import json
import os
import time
from kpi_enhanced import (
    EnhancedAgentBenchmarkKPI,
    run_agent_pipeline,
    get_video_duration,
    tolerance_for_duration
)

SEPARATOR = "=" * 120
SUB_SEP = "-" * 120

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    video_path = sys.argv[1]
    gt_path = sys.argv[2] if len(sys.argv) > 2 else "ground_truth.json"
    gt_video_id = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(video_path)

    print(f"\n{SEPARATOR}")
    print(f"🎥 VİDEO TESTİ: {video_path}")
    print(SEPARATOR)

    # 1. Ground Truth'u Yükle
    gt_entry = {}
    if gt_path and os.path.exists(gt_path):
        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_list = json.load(f)
            gt_entry = next((item for item in gt_list if item.get("video_id") == gt_video_id), {})
            if not gt_entry:
                print(f"⚠️ '{gt_video_id}' ID'li video {gt_path} içinde bulunamadı! Sadece ham model çıktısı gösterilecek.")
        except Exception as e:
            print(f"⚠️ GT dosyası okunamadı: {e}")
    else:
        print("ℹ️ Ground Truth bulunamadı/verilmedi. Benchmark yapılmayacak, ham model çıktısı gösterilecek.")

    # 2. Pipeline'ı Çalıştır ve KPI Ölçümünü Yap
    print("⏳ AI Ajanı ve KPI Pipeline çalışıyor (Lütfen bekleyin)...\n")
    
    gt_duration = float(gt_entry.get("duration_sec", 0) or 0)
    duration = get_video_duration(video_path, fallback=gt_duration) or gt_duration
    
    if not gt_entry:
        # Memory sıfırla — her test öncesi temiz başlangıç
        import json as _json, os as _os
        _mem_path = "memory/memory.json"
        if _os.path.exists(_mem_path):
            with open(_mem_path, "w") as _f:
                _json.dump({"event_history": [], "risk_history": [], "tool_history": []}, _f)
                
        start_time = time.perf_counter()
        raw_output = run_agent_pipeline(video_path)
        elapsed = time.perf_counter() - start_time
        print(f"{SEPARATOR}\n📦 HAM MODEL ÇIKTISI (Süre: {elapsed:.2f} sn)\n{SEPARATOR}")
        print(f"Risk Seviyesi : {raw_output.get('risk_level')}")
        print(f"Özet          : {raw_output.get('summary')}")
        print(f"Araçlar       : {raw_output.get('actions')}")
        print(f"Bulunan Olaylar:")
        for ev in raw_output.get('events', []):
            print(f"  - [{ev.get('time')}] {ev.get('event')}")
        return

    bench = EnhancedAgentBenchmarkKPI(drop_synthetic_events=False)
    res = bench.run_benchmark(run_agent_pipeline, video_path, gt_entry, duration)

    acc = res["accuracy"]
    perf = res["performance"]
    layer = res["layer_analysis"]
    tools_info = layer["tool_selection"]
    
    # 3. Terminal Raporunu Bas
    print(SEPARATOR)
    print("📊 KPI SKOR ÖZETİ (TEKNOFEST TYDA ŞARTNAME UYUMLU)")
    print(SEPARATOR)
    print(f"  • Genel Sistem Skoru     : {layer['overall_score']:.3f}")
    print(f"  • Olay Tespit (F1)       : {acc['event_f1_score']:.3f} (Precision: {acc['event_precision']:.2f}, Recall: {acc['event_recall']:.2f})")
    if acc.get("critical_recall") is not None:
        print(f"  • Kritik Olay Yakalama   : {acc['critical_recall']:.3f}")
    print(f"  • Özet Kalitesi          : {acc['summary_score'] or 0.0:.3f}")
    
    # Tool Göstergeleri: Exact Match ve IoU
    exact_str = "✅ Tam Eşleşti (1.0)" if tools_info.get("tool_exact_match") == 1.0 else "❌ Farklılık Var (0.0)"
    iou_val = acc['tool_jaccard_score'] if acc['tool_jaccard_score'] is not None else 0.0
    print(f"  • Araç Tam Eşleşme (Exact): {exact_str}")
    print(f"  • Araç Önerisi (IoU/Jaccard): {iou_val:.3f} (Recall: {tools_info.get('tool_recall', 0.0):.2f}, Precision: {tools_info.get('tool_precision', 0.0):.2f})")
    print(f"  • Risk Doğruluğu         : {'✅ Doğru' if acc['risk_accuracy'] == 1.0 else '❌ Yanlış' if acc['risk_accuracy'] == 0.0 else '⚠️ Ölçülemedi'}")
    print(f"  • İşlem Süresi           : {perf['total_processing_time_sec']:.2f} sn (RTF: {perf['real_time_factor'] or 0.0:.2f})")
    
    # --- ARAÇ DETAYLARI ---
    print(f"\n{SUB_SEP}")
    print("🔧 ARAÇ (TOOL) ANALİZİ DETAYI")
    print(SUB_SEP)
    expected = set(tools_info["expected_tools"])
    detected = set(tools_info["detected_tools"])
    
    intersection = expected & detected
    missed = expected - detected
    extra = detected - expected
    
    print(f"  Beklenen : {expected or '(yok)'}")
    print(f"  Üretilen : {detected or '(yok)'}")
    
    if intersection:
        print("\n  ✅ DOĞRU ÇAĞRILANLAR:")
        for i in intersection: print(f"    - {i}")
    if missed:
        print("\n  📉 EKSİK ARAÇLAR (Modelin unuttuğu):")
        for m in missed: print(f"    - {m}")
    if extra:
        print("\n  📈 FAZLADAN ARAÇLAR (Modelin uydurduğu / Hallucination):")
        for e in extra: print(f"    - {e}")

    # --- ÖZET DETAYLARI ---
    print(f"\n{SUB_SEP}")
    print("📝 ÖZET KALİTESİ DETAYI")
    print(SUB_SEP)
    print(f"  GT Özeti    : {acc.get('gt_summary_text', '')}")
    print(f"  Model Özeti : {acc.get('summary_text', '')}")
    
    # --- OLAY DETAYLARI ---
    print(f"\n{SUB_SEP}")
    print("⏱️ ZAMANSAL OLAY EŞLEŞME DETAYI")
    print(SUB_SEP)
    events = layer["event_analysis"]
    print(f"  Tolerans Penceresi: ±{events['tolerance_sec']} sn\n")
    
    if events["matches"]:
        print("  ✅ EŞLEŞEN OLAYLAR (True Positives):")
        for m in events["matches"]:
            print(f"    • GT [{m['gt_interval'][0]}-{m['gt_interval'][1]}] : {m['gt']}")
            print(f"      MD [{m['pred_interval'][0]}-{m['pred_interval'][1]}] : {m['pred']}")
            print(f"      (IoM: {m['iom']:.2f}, Metin Benzerliği: {m['text_sim']:.2f})\n")
            
    if events["missed_gt"]:
        print("  📉 KAÇIRILAN OLAYLAR (False Negatives):")
        for m in events["missed_gt"]:
            iv_str = f"[{m['interval'][0]}-{m['interval'][1]}]" if m['interval'] else "[Zaman Yok]"
            print(f"    • {iv_str} {m['text']} {'(🚨 KRİTİK)' if m['critical'] else ''}")
            
    if events["false_positives"]:
        print("\n  📈 YANLIŞ ALARMLAR / UYDURMALAR (False Positives):")
        for fp in events["false_positives"]:
            iv_str = f"[{fp['interval'][0]}-{fp['interval'][1]}]" if fp['interval'] else "[Zaman Yok]"
            print(f"    • {iv_str} {fp['text']}")

    print(f"\n{SEPARATOR}\n")

if __name__ == "__main__":
    main()