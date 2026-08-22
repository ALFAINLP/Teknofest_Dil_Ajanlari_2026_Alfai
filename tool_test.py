"""
debug_tool_match.py
-------------------
KPI benchmark'ı çalıştırmadan tek bir video için tool eşleşmesini gösterir.

Kullanım:
    python debug_tool_match.py <video_path> [ground_truth_json_path] [ground_truth_video_id]

Örnekler:
    # Sadece pipeline çıktısını gör (ground truth olmadan):
    python debug_tool_match.py Videos/test.mp4

    # Ground truth ile karşılaştır:
    python debug_tool_match.py Videos/test.mp4 ground_truth.json test.mp4
"""

import sys
import json
import re
import time
from typing import Set, List, Any


# ── Ayarlar ──────────────────────────────────────────────────────────────────

VALID_TOOLS = {
    "trigger_operator_alert",
    "get_personnel_on_duty",
    "assess_environmental_hazards",
    "query_previous_incidents",
    "export_to_incident_management_system",
    "detect_motion_density"
    }

SEPARATOR = "=" * 60


# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────

def extract_tool_names(action_list: List[Any]) -> Set[str]:
    """kpi.py'daki _extract_tool_names ile birebir aynı mantık."""
    extracted = set()
    for item in action_list:
        item_str = str(item).lower()
        for vt in VALID_TOOLS:
            if vt in item_str:
                extracted.add(vt)
    return extracted


def jaccard(a: Set[str], b: Set[str]) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def exact_match(a: Set[str], b: Set[str]) -> bool:
    return a == b


def run_pipeline(video_path: str) -> dict:
    """
    kpi.py'daki run_agent_pipeline ile birebir aynı — sadece buraya kopyalandı
    ki bu script bağımsız çalışabilsin.
    """
    from agent.graph import app

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

    t0 = time.perf_counter()
    final_output = app.invoke(initial_state)
    elapsed = round(time.perf_counter() - t0, 2)

    # JSON çıktısını çözümle
    final_json = final_output.get("final_json_output", "")
    parsed_report = {}
    if isinstance(final_json, str) and final_json.strip():
        try:
            cleaned = re.sub(r"^```json\s*|\s*```$", "", final_json.strip(), flags=re.MULTILINE)
            parsed_report = json.loads(cleaned)
        except Exception:
            pass
    elif isinstance(final_json, dict):
        parsed_report = final_json

    detected_risk = parsed_report.get("risk", "Düşük")
    detected_events = parsed_report.get("events", final_output.get("detected_events", []))

    # Tool eşleşmesi — kpi.py ile aynı mantık
    executed_tools = set(final_output.get("actions_recommended", []))
    for item in final_output.get("tool_history", []):
        if isinstance(item, dict) and "tool" in item:
            executed_tools.add(item["tool"])

    return {
        "risk_level": detected_risk,
        "events": detected_events,
        "actions": list(executed_tools),
        # Debug için ekstra bilgi
        "_raw_actions_recommended": final_output.get("actions_recommended", []),
        "_raw_tool_history": [
            item.get("tool") for item in final_output.get("tool_history", [])
            if isinstance(item, dict) and "tool" in item
        ],
        "_elapsed_sec": elapsed,
    }


# ── Ana fonksiyon ─────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    video_path = sys.argv[1]
    gt_path = sys.argv[2] if len(sys.argv) > 2 else None
    gt_video_id = sys.argv[3] if len(sys.argv) > 3 else None

    print(f"\n{SEPARATOR}")
    print(f"🎥 Video: {video_path}")
    print(SEPARATOR)

    # Pipeline'ı çalıştır
    print("⏳ Pipeline çalışıyor...")
    result = run_pipeline(video_path)

    # ── Pipeline çıktısı ──────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("📦 PIPELINE ÇIKTISI")
    print(SEPARATOR)
    print(f"  Risk seviyesi : {result['risk_level']}")
    print(f"  Olay sayısı   : {len(result['events'])}")
    print(f"  Süre          : {result['_elapsed_sec']} sn")

    print(f"\n  actions_recommended (ham):")
    for item in result["_raw_actions_recommended"]:
        print(f"    • {item}")

    print(f"\n  tool_history (araç adları):")
    for item in result["_raw_tool_history"]:
        print(f"    • {item}")

    # ── Tool eşleşme analizi ──────────────────────────────────────────────
    pred_tools = extract_tool_names(result["actions"])
    print(f"\n{SEPARATOR}")
    print("🔧 TOOL EŞLEŞMESİ")
    print(SEPARATOR)
    print(f"  extract_tool_names() girdisi : {result['actions']}")
    print(f"  Tanınan pred_tools           : {pred_tools or '(boş set)'}")

    # Ground truth varsa karşılaştır
    if gt_path and gt_video_id:
        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_list = json.load(f)

            gt_entry = next(
                (item for item in gt_list if item.get("video_id") == gt_video_id),
                None
            )

            if gt_entry is None:
                print(f"\n⚠️  '{gt_video_id}' ground_truth.json'da bulunamadı.")
                print(f"  Mevcut video_id'ler: {[x.get('video_id') for x in gt_list]}")
            else:
                true_tools = extract_tool_names(gt_entry.get("actions", []))
                intersection = pred_tools & true_tools
                union = pred_tools | true_tools
                j_score = jaccard(pred_tools, true_tools)
                em = exact_match(pred_tools, true_tools)

                print(f"\n{SEPARATOR}")
                print("📊 GROUND TRUTH KARŞILAŞTIRMASI")
                print(SEPARATOR)
                print(f"  Beklenen tools   : {true_tools or '(boş set)'}")
                print(f"  Tahmin tools     : {pred_tools or '(boş set)'}")
                print(f"  Kesişim (✓)      : {intersection or '(yok)'}")
                print(f"  Yalnız pred (✗)  : {pred_tools - true_tools or '(yok)'}")
                print(f"  Yalnız GT   (✗)  : {true_tools - pred_tools or '(yok)'}")
                print(f"  Jaccard (IoU)    : {j_score:.3f}")
                print(f"  Exact Match      : {'✅ Evet' if em else '❌ Hayır'}")

                # Sorun tespiti
                # ── Detaylı Exact Match ve Sorun Logu ──────────────────────
                print(f"\n{SEPARATOR}")
                print("🔍 DETAYLI EXACT MATCH ANALİZİ")
                print(SEPARATOR)
                
                if em:
                    print("  ✅ EXACT MATCH BAŞARILI! Sistem beklenen tüm araçları eksiksiz çağırdı.")
                else:
                    print("  ❌ EXACT MATCH BAŞARISIZ! Nedenleri:")
                    
                    # Eksik Araçlar (Olması gereken ama bulunamayanlar)
                    missed = true_tools - pred_tools
                    if missed:
                        print(f"\n     📉 EKSİK ARAÇLAR (GT'de var, sistem çağırmadı):")
                        for m in missed:
                            print(f"        - {m}")
                            if m == "get_personnel_on_duty":
                                print("          (İpucu: Olayda işçi var ama regex/prompt bunu tetikleyemedi)")
                            elif m == "export_to_incident_management_system":
                                print("          (İpucu: Risk Kritik/Yüksek hesaplanamadığı için CMS'ye aktarılamadı)")
                    
                    # Fazladan Araçlar (Olmaması gereken ama uydurulanlar)
                    extra = pred_tools - true_tools
                    if extra:
                        print(f"\n     📈 FAZLADAN ARAÇLAR (GT'de yok, sistem gereksiz çağırdı):")
                        for e in extra:
                            print(f"        - {e}")
                            if e == "log_kpi_metrics":
                                print("          (İpucu: Olay Kritik olmasına rağmen risk 'Düşük' sanıldığı için rutin log atılmış)")
                            elif e == "submit_final_decision":
                                print("          (İpucu: Bu sistem aracıdır, VALID_TOOLS listesinden kaldırılmalıdır)")

                    # Ortak Başarılar
                    if intersection:
                        print(f"\n     🎯 DOĞRU ÇAĞRILANLAR (Ortak):")
                        for i in intersection:
                            print(f"        - {i}")

        except FileNotFoundError:
            print(f"\n⚠️  Ground truth dosyası bulunamadı: {gt_path}")
    else:
        print("\n  ℹ️  Ground truth verilmedi, karşılaştırma yapılmıyor.")
        print("     Kullanım: python debug_tool_match.py <video> <gt.json> <video_id>")

    print(f"\n{SEPARATOR}\n")


if __name__ == "__main__":
    main()