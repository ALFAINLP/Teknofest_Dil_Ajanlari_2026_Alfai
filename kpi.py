#!/usr/bin/env python3
"""
diagnose_summary.py — "Özet kalitesi neden 0.386?" sorusunu veriye dayandırır.

diagnose_kpi.py'nin OLAY katmanı için yaptığını ÖZET katmanı için yapar.
Pipeline'ı yeniden çalıştırmaz; mevcut benchmark çıktısını + ground_truth'u okur.

Yanıtladığı sorular
-------------------
1. Skor precision-bound mu, recall-bound mu?
      recall düşük  -> model GT'deki içeriği kapsamıyor (kısa/eksik yazıyor)
      precision düşük -> model GT'de olmayan şey ekliyor (uzun/dağınık yazıyor)
2. Uzunluk oranı (pred/gt) ne? F skoru uzunlukla korele mi?
3. Skorun ne kadarı MORFOLOJİ VERGİSİ? (gövdeleme açınca ne kadar sıçrıyor)
4. Kaç videoda özet boş / fallback'e düşmüş? (graph.py'deki 240-char fallback)
5. Hangi GT terimleri sistematik olarak kaçırılıyor? (prompt terminolojisi için)
6. En kötü 10 videoda GT ve Pred yan yana neye benziyor?

Kullanım
--------
    python diagnose_summary.py
    python diagnose_summary.py --input benchmark_results_enhanced.json \
                               --gt ground_truth.json --top 10
    python diagnose_summary.py --dump-schema     # alan adları tutmuyorsa BUNU çalıştır
    python diagnose_summary.py --csv summary_diag.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ----------------------------------------------------------------------------
# kpi_enhanced varsa onun işlemcisini de yanına koyarız (karşılaştırma için)
# ----------------------------------------------------------------------------
try:
    from kpi_enhanced import TurkishTextProcessor  # type: ignore
    HAS_KPI = True
except Exception:
    TurkishTextProcessor = None  # type: ignore
    HAS_KPI = False


# ----------------------------------------------------------------------------
# Türkçe metin işleme
# ----------------------------------------------------------------------------
TR_STOPWORDS = {
    "acaba", "ama", "ancak", "artık", "asla", "aslında", "az", "bazı", "belki",
    "ben", "beri", "bile", "bir", "biraz", "birçok", "biri", "birkaç", "birşey",
    "biz", "bu", "buna", "bunda", "bundan", "bunu", "bunun", "burada", "çok",
    "çünkü", "da", "daha", "de", "defa", "değil", "diğer", "diye", "dolayı",
    "eğer", "en", "fakat", "gibi", "göre", "hala", "hangi", "hatta", "hem",
    "henüz", "hep", "hepsi", "her", "herhangi", "hiç", "için", "ile", "ise",
    "işte", "kadar", "karşın", "kendi", "ki", "kim", "mı", "mi", "mu", "mü",
    "nasıl", "ne", "neden", "nerede", "niçin", "o", "olarak", "olan", "olduğu",
    "olup", "ona", "ondan", "onlar", "onun", "önce", "pek", "sadece", "sanki",
    "şey", "şu", "sonra", "tarafından", "tüm", "üzere", "ve", "veya", "ya",
    "yani", "yine", "yoksa", "zaten", "ayrıca", "bunlar", "böyle", "şekilde",
    "durumu", "durum", "adlı", "isimli", "içinde", "üzerinde", "arasında",
}

_TOKEN_RE = re.compile(r"[a-zçğıöşü0-9]+")


def tr_lower(s: str) -> str:
    """Türkçe'ye duyarlı küçültme (İ->i, I->ı)."""
    return s.replace("İ", "i").replace("I", "ı").lower()


def tokenize(s: str) -> List[str]:
    if not s:
        return []
    return _TOKEN_RE.findall(tr_lower(str(s)))


def drop_stop(tokens: Iterable[str]) -> List[str]:
    return [t for t in tokens if t not in TR_STOPWORDS and len(t) > 1]


def stem5(tokens: Iterable[str], n: int = 5) -> List[str]:
    """Sabit uzunlukta prefix gövdeleme. Türkçe için ucuz ve etkili yaklaşım."""
    return [t[:n] for t in tokens]


def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 1:
        return [(t,) for t in tokens]
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def prf(pred: List[Any], gt: List[Any]) -> Tuple[float, float, float]:
    """Clipped-count precision / recall / F1."""
    if not pred or not gt:
        return 0.0, 0.0, 0.0
    pc, gc = Counter(pred), Counter(gt)
    overlap = sum(min(c, gc[k]) for k, c in pc.items())
    p = overlap / len(pred)
    r = overlap / len(gt)
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l(pred: List[str], gt: List[str]) -> Tuple[float, float, float]:
    if not pred or not gt:
        return 0.0, 0.0, 0.0
    l = lcs_len(pred, gt)
    p, r = l / len(pred), l / len(gt)
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


# ----------------------------------------------------------------------------
# Şema keşfi — alan adlarını sabit varsaymıyoruz
# ----------------------------------------------------------------------------
PRED_SUMMARY_KEYS = [
    "pred_summary", "predicted_summary", "summary_pred", "prediction_summary",
    "generated_summary", "model_summary", "pred_text", "output_summary",
    "summary",
]
GT_SUMMARY_KEYS = [
    "gt_summary", "ground_truth_summary", "reference_summary", "ref_summary",
    "summary_gt", "expected_summary", "summary", "ozet", "özet", "description",
]
SUMMARY_SCORE_KEYS = [
    "summary_quality", "summary_quality_score", "summary_score",
    "summary_similarity", "summary_f1", "rouge_score", "summary_rouge",
]
VIDEO_ID_KEYS = ["video_id", "video", "id", "name", "file_name", "filename"]


def walk(obj: Any, path: str = "") -> Iterable[Tuple[str, str, Any]]:
    """(path, key, value) üzerinde derinlemesine gezinti."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            yield p, str(k), v
            yield from walk(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:3]):  # ilk 3 eleman yeter
            p = f"{path}[{i}]"
            yield from walk(v, p)


def find_str(obj: Any, candidates: List[str], min_len: int = 15) -> Tuple[Optional[str], str]:
    """Aday anahtar adlarına göre ilk anlamlı string'i bul. (değer, bulunduğu yol)"""
    found = list(walk(obj))
    for cand in candidates:  # öncelik sırası korunur
        for path, key, val in found:
            if key.lower() == cand and isinstance(val, str) and len(val.strip()) >= min_len:
                return val.strip(), path
    return None, ""


def find_num(obj: Any, candidates: List[str]) -> Tuple[Optional[float], str]:
    found = list(walk(obj))
    for cand in candidates:
        for path, key, val in found:
            if key.lower() == cand and isinstance(val, (int, float)):
                return float(val), path
    return None, ""


def find_id(row: Dict[str, Any]) -> str:
    for k in VIDEO_ID_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v:
            return v
    return "?"


def dump_schema(row: Dict[str, Any]) -> None:
    print("\n" + "=" * 74)
    print("İLK SONUÇ SATIRININ ANAHTAR HARİTASI")
    print("=" * 74)
    print("Aşağıda 'summary' / 'ozet' geçen ne varsa listeleniyor.")
    print("Doğru alanı görüp script'in üstündeki *_KEYS listelerine ekleyin.\n")
    hits = 0
    for path, key, val in walk(row):
        kl = key.lower()
        if "summary" in kl or "ozet" in kl or "özet" in kl or "rouge" in kl:
            preview = str(val)[:70].replace("\n", " ")
            t = type(val).__name__
            print(f"  {path:<58} ({t}) {preview}")
            hits += 1
    if not hits:
        print("  ⚠️  'summary' içeren HİÇBİR anahtar yok.")
        print("      Özet skoru muhtemelen benchmark JSON'una hiç yazılmıyor.")
        print("      Tüm üst seviye anahtarlar:")
        for k in row:
            print(f"        - {k}")
    print()


# ----------------------------------------------------------------------------
# Ground truth yükleme
# ----------------------------------------------------------------------------
def load_gt(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}

    if isinstance(data, dict):
        for wrapper in ("videos", "data", "items", "ground_truth", "results"):
            if wrapper in data and isinstance(data[wrapper], (list, dict)):
                data = data[wrapper]
                break

    index: Dict[str, Any] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            index[str(k)] = v
    elif isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                index[find_id(entry)] = entry
    return index


def norm_id(s: str) -> str:
    """video_1.mp4 / video_1 / VIDEO_1 aynı kabul edilsin."""
    s = tr_lower(str(s)).strip()
    for ext in (".mp4", ".avi", ".mov", ".mkv", ".json"):
        if s.endswith(ext):
            s = s[: -len(ext)]
    return s


def lookup_gt(gt_index: Dict[str, Any], vid: str) -> Optional[Any]:
    if vid in gt_index:
        return gt_index[vid]
    nv = norm_id(vid)
    for k, v in gt_index.items():
        if norm_id(k) == nv:
            return v
    return None


# ----------------------------------------------------------------------------
# Video başına analiz
# ----------------------------------------------------------------------------
def analyze_pair(gt_text: str, pred_text: str) -> Dict[str, Any]:
    gt_raw, pred_raw = tokenize(gt_text), tokenize(pred_text)
    gt_s, pred_s = drop_stop(gt_raw), drop_stop(pred_raw)
    gt_st, pred_st = stem5(gt_s), stem5(pred_s)

    p_raw, r_raw, f_raw = prf(pred_raw, gt_raw)
    p_st, r_st, f_st = prf(pred_st, gt_st)
    p_b, r_b, f_b = prf(ngrams(pred_st, 2), ngrams(gt_st, 2))
    p_l, r_l, f_l = rouge_l(pred_st, gt_st)

    missed = [t for t in set(gt_st) if t not in set(pred_st)]
    extra = [t for t in set(pred_st) if t not in set(gt_st)]

    return {
        "gt_len": len(gt_raw), "pred_len": len(pred_raw),
        "len_ratio": (len(pred_raw) / len(gt_raw)) if gt_raw else 0.0,
        "p_raw": p_raw, "r_raw": r_raw, "f_raw": f_raw,
        "p": p_st, "r": r_st, "f": f_st,
        "f_bigram": f_b, "f_rougeL": f_l, "p_rougeL": p_l, "r_rougeL": r_l,
        "combo": 0.4 * f_st + 0.2 * f_b + 0.4 * f_l,
        "missed_terms": missed, "extra_terms": extra,
        "gt_text": gt_text, "pred_text": pred_text,
    }


def bar(v: float, width: int = 20) -> str:
    return "█" * int(max(0.0, min(1.0, v)) * width)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="benchmark_results_enhanced.json")
    ap.add_argument("--gt", default="ground_truth.json")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--csv", default=None, help="Video başına metrikleri CSV'ye yaz")
    ap.add_argument("--dump-schema", action="store_true",
                    help="Alan adları tutmuyorsa mevcut anahtarları listele ve çık")
    args = ap.parse_args()

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ {args.input} bulunamadı.")
        return 1

    results = data["results"] if isinstance(data, dict) and "results" in data else data
    if not results:
        print("❌ Sonuç listesi boş.")
        return 1

    if args.dump_schema:
        dump_schema(results[0])
        return 0

    gt_index = load_gt(args.gt)

    rows: List[Dict[str, Any]] = []
    no_gt: List[str] = []
    no_pred: List[str] = []
    reported_scores: List[float] = []
    pred_path_seen = gt_path_seen = ""

    for row in results:
        vid = find_id(row)

        pred_text, pp = find_str(row, PRED_SUMMARY_KEYS)
        if pp:
            pred_path_seen = pp

        gt_text, gp = (None, "")
        gt_text, gp = find_str(row, ["gt_summary", "ground_truth_summary",
                                     "reference_summary", "ref_summary"])
        if not gt_text:
            entry = lookup_gt(gt_index, vid)
            if entry is not None:
                gt_text, gp = find_str(entry, GT_SUMMARY_KEYS)
        if gp:
            gt_path_seen = gp

        score, _ = find_num(row, SUMMARY_SCORE_KEYS)
        if score is not None:
            reported_scores.append(score)

        if not gt_text:
            no_gt.append(vid)
            continue
        if not pred_text:
            no_pred.append(vid)
            rows.append({"video": vid, "empty_pred": True,
                         **analyze_pair(gt_text, "")})
            continue

        rows.append({"video": vid, "empty_pred": False,
                     "reported": score, **analyze_pair(gt_text, pred_text)})

    print("\n" + "=" * 74)
    print("VERİ KAYNAĞI")
    print("=" * 74)
    print(f"  Sonuç satırı           : {len(results)}")
    print(f"  Pred özet bulunan yol  : {pred_path_seen or '— BULUNAMADI —'}")
    print(f"  GT özet bulunan yol    : {gt_path_seen or '— BULUNAMADI —'}")
    print(f"  kpi_enhanced import    : {'evet' if HAS_KPI else 'hayır'}")
    if reported_scores:
        m = sum(reported_scores) / len(reported_scores)
        print(f"  Benchmark'ın kendi özet skoru (ort.): {m:.3f}")
    if no_gt:
        print(f"  ⚠️  GT özeti bulunamayan {len(no_gt)} video: {', '.join(no_gt[:6])}"
              + (" ..." if len(no_gt) > 6 else ""))
    if not rows:
        print("\n❌ Karşılaştırılabilir tek bir çift bile kurulamadı.")
        print("   --dump-schema ile alan adlarına bakın.")
        return 1

    n = len(rows)
    avg = lambda k: sum(r[k] for r in rows) / n  # noqa: E731

    # ---------------- 1. PRECISION Mİ RECALL Mİ ----------------
    print("\n" + "=" * 74)
    print("1) SKOR PRECISION-BOUND MU, RECALL-BOUND MU?")
    print("=" * 74)
    print(f"  {'':<24}{'precision':>11}{'recall':>10}{'F1':>10}")
    print(f"  {'ham token':<24}{avg('p_raw'):>11.3f}{avg('r_raw'):>10.3f}{avg('f_raw'):>10.3f}")
    print(f"  {'stopword+gövde(5)':<24}{avg('p'):>11.3f}{avg('r'):>10.3f}{avg('f'):>10.3f}")
    print(f"  {'ROUGE-L (gövdeli)':<24}{avg('p_rougeL'):>11.3f}{avg('r_rougeL'):>10.3f}{avg('f_rougeL'):>10.3f}")
    print(f"  {'bigram (gövdeli)':<24}{'':>11}{'':>10}{avg('f_bigram'):>10.3f}")
    print(f"\n  Önerilen bileşik (0.4·R1 + 0.2·R2 + 0.4·RL): {avg('combo'):.3f}")

    p_m, r_m = avg("p"), avg("r")
    print("\n  TEŞHİS:")
    if r_m < p_m - 0.08:
        print("  → RECALL-BOUND. Model GT'deki içeriğin büyük kısmını hiç yazmıyor.")
        print("    Müdahale: özeti uzat, olay zincirini (başlangıç→gelişim→kritik→sonuç)")
        print("    zorunlu kıl. Eşikle veya ağırlıkla oynamak bunu düzeltmez.")
    elif p_m < r_m - 0.08:
        print("  → PRECISION-BOUND. Model GT'de olmayan çok şey yazıyor.")
        print("    Müdahale: özeti kısalt, kare kare betimlemeyi yasakla,")
        print("    uzunluk tavanı koy. İçerik zaten yakalanmış.")
    else:
        print("  → DENGELİ. Tek yönlü bir hata yok; sorun ifade/terminoloji farkında.")
        print("    Aşağıdaki 'kaçırılan terimler' listesi asıl yol gösterici.")

    # ---------------- 2. MORFOLOJİ VERGİSİ ----------------
    delta = avg("f") - avg("f_raw")
    print("\n" + "=" * 74)
    print("2) MORFOLOJİ VERGİSİ (gövdelemenin tek başına getirisi)")
    print("=" * 74)
    print(f"  Ham F1        : {avg('f_raw'):.3f}")
    print(f"  Gövdeli F1    : {avg('f'):.3f}")
    print(f"  Fark          : {delta:+.3f}")
    if delta >= 0.08:
        print("\n  → Skorunuzun kayda değer bir kısmı MODEL HATASI DEĞİL, ölçüm artefaktı.")
        print("    kpi_enhanced içindeki özet tokenizasyonuna gövdeleme ekleyin;")
        print("    bu tek satırlık değişiklik model dokunmadan kazanç verir.")
    else:
        print("\n  → Gövdeleme az getiriyor. Sorun çekim ekleri değil, İÇERİK farkı.")
        print("    Yani model gerçekten farklı şeylerden bahsediyor.")

    # ---------------- 2b. GÖVDE UZUNLUĞU TARAMASI ----------------
    print("\n" + "=" * 74)
    print("2b) GÖVDE UZUNLUĞU DUYARLILIĞI (hangi prefix en iyi?)")
    print("=" * 74)
    print(f"  {'prefix':<10}{'precision':>11}{'recall':>10}{'F1':>10}")
    best_k, best_f = None, -1.0
    for k in (3, 4, 5, 6, 7, 99):
        acc = [0.0, 0.0, 0.0]
        for r in rows:
            g = stem5(drop_stop(tokenize(r["gt_text"])), k)
            p = stem5(drop_stop(tokenize(r["pred_text"])), k)
            vals = prf(p, g)
            acc = [a + b for a, b in zip(acc, vals)]
        p_, r_, f_ = [a / n for a in acc]
        label = "gövdesiz" if k == 99 else f"ilk {k}"
        mark = ""
        if f_ > best_f:
            best_f, best_k, mark = f_, k, ""
        print(f"  {label:<10}{p_:>11.3f}{r_:>10.3f}{f_:>10.3f}  {bar(f_)}{mark}")
    lbl = "gövdesiz" if best_k == 99 else f"prefix-{best_k}"
    print(f"\n  → En iyi: {lbl} (F1={best_f:.3f}). kpi_enhanced'de bunu kullanın.")
    print("  ⚠️  Çok kısa prefix farklı kelimeleri birleştirip skoru YAPAY şişirir.")
    print("     3-4 arası kazanç büyükse şüphelenin; 5-6 genelde güvenli banttır.")

    # ---------------- 3. UZUNLUK ----------------
    print("\n" + "=" * 74)
    print("3) UZUNLUK ANALİZİ")
    print("=" * 74)
    gl = sorted(r["gt_len"] for r in rows)
    pl = sorted(r["pred_len"] for r in rows)
    med = lambda a: a[len(a) // 2]  # noqa: E731
    print(f"  GT   token: min {gl[0]:>3} | medyan {med(gl):>3} | max {gl[-1]:>3}")
    print(f"  Pred token: min {pl[0]:>3} | medyan {med(pl):>3} | max {pl[-1]:>3}")
    print(f"  Ortalama uzunluk oranı (pred/gt): {avg('len_ratio'):.2f}")
    print(f"\n  → Prompt'a yazılacak hedef: yaklaşık {int(med(gl) * 0.85)}–{int(med(gl) * 1.15)} kelime.\n")

    buckets = [("<0.50 (çok kısa)", 0.0, 0.5), ("0.50-0.80", 0.5, 0.8),
               ("0.80-1.25 (ideal)", 0.8, 1.25), ("1.25-2.00", 1.25, 2.0),
               (">2.00 (çok uzun)", 2.0, 9e9)]
    print(f"  {'uzunluk oranı':<22}{'video':>7}{'ort. F1':>10}")
    for label, lo, hi in buckets:
        sub = [r for r in rows if lo <= r["len_ratio"] < hi]
        if not sub:
            continue
        f = sum(r["f"] for r in sub) / len(sub)
        print(f"  {label:<22}{len(sub):>7}{f:>10.3f}  {bar(f)}")
    print("\n  Eğer 'ideal' bandındaki F1 diğerlerinden belirgin yüksekse,")
    print("  kaybınızın bir kısmı doğrudan UZUNLUK KALİBRASYONU sorunudur.")

    # ---------------- 4. BOŞ / FALLBACK ----------------
    empty = [r for r in rows if r["empty_pred"] or r["pred_len"] == 0]
    tiny = [r for r in rows if not r["empty_pred"] and 0 < r["pred_len"] < 8]
    print("\n" + "=" * 74)
    print("4) BOŞ / FALLBACK ÖZETLER")
    print("=" * 74)
    print(f"  Tamamen boş özet : {len(empty)} video")
    print(f"  8 token'dan kısa : {len(tiny)} video")
    if empty or tiny:
        bad = [r["video"] for r in (empty + tiny)][:10]
        print(f"  → {', '.join(bad)}")
        loss = sum(r["f"] for r in rows) / n
        bad_ids = {r["video"] for r in empty} | {r["video"] for r in tiny}
        without = [r for r in rows if r["video"] not in bad_ids]
        if without:
            f_wo = sum(r["f"] for r in without) / len(without)
            print(f"\n  Bunlar hariç ort. F1 = {f_wo:.3f} (hepsiyle {loss:.3f})")
            print(f"  Sadece bu {len(empty) + len(tiny)} videoyu düzeltmek "
                  f"+{f_wo - loss:.3f} getirir.")
        print("\n  NOT: graph.py'de video_summary boşsa ilk 2 olay birleştirilip")
        print("       240 karakterde kesiliyor. Bu videolarda vision özeti hiç")
        print("       üretilmemiş olabilir — önce ORAYA bakın.")
    else:
        print("  ✓ Boş özet yok; kayıp kalite kaynaklı.")

    # ---------------- 5. TERMİNOLOJİ ----------------
    missed_c: Counter = Counter()
    extra_c: Counter = Counter()
    for r in rows:
        missed_c.update(set(r["missed_terms"]))
        extra_c.update(set(r["extra_terms"]))

    print("\n" + "=" * 74)
    print("5) TERMİNOLOJİ FARKI (prompt mühendisliği için)")
    print("=" * 74)
    print("  GT'de olup tahminde OLMAYAN gövdeler (kaç videoda):")
    for term, c in missed_c.most_common(20):
        if c >= max(2, n // 10):
            print(f"    {term:<14} {c:>3} video  {bar(c / n, 14)}")
    print("\n  Tahminde olup GT'de OLMAYAN gövdeler:")
    for term, c in extra_c.most_common(20):
        if c >= max(2, n // 10):
            print(f"    {term:<14} {c:>3} video  {bar(c / n, 14)}")
    print("\n  → İlk liste: prompt'a 'tercih edilen terminoloji' olarak eklenecek adaylar.")
    print("  → İkinci liste: modelin fazladan söylediği şeyler (dolgu mu, halüsinasyon mu?).")
    print("  ⚠️  45 videoya aşırı uyum riski var. Sadece GENEL terimleri alın,")
    print("     videoya özgü nesne isimlerini DEĞİL.")

    # ---------------- 6. VIDEO TABLOSU ----------------
    print("\n" + "=" * 74)
    print(f"6) EN KÖTÜ {args.top} VİDEO")
    print("=" * 74)
    print(f"{'video':<24}{'P':>7}{'R':>7}{'F1':>7}{'RL':>7}"
          f"{'gt':>5}{'pr':>5}{'oran':>7}")
    worst = sorted(rows, key=lambda x: x["f"])[:args.top]
    for r in worst:
        print(f"{r['video'][:23]:<24}{r['p']:>7.2f}{r['r']:>7.2f}{r['f']:>7.2f}"
              f"{r['f_rougeL']:>7.2f}{r['gt_len']:>5}{r['pred_len']:>5}"
              f"{r['len_ratio']:>7.2f}")

    print("\n" + "=" * 74)
    print("7) YAN YANA — EN KÖTÜLER")
    print("=" * 74)
    for r in worst:
        print(f"\n─── {r['video']}   F1={r['f']:.2f}  P={r['p']:.2f} R={r['r']:.2f}"
              f"  oran={r['len_ratio']:.2f}")
        print(f"  GT   ({r['gt_len']:>3} tok): {r['gt_text'][:300]}")
        print(f"  PRED ({r['pred_len']:>3} tok): {r['pred_text'][:300] or '— BOŞ —'}")
        miss = ", ".join(sorted(r["missed_terms"])[:12])
        print(f"  kaçan: {miss}")

    # ---------------- CSV ----------------
    if args.csv:
        cols = ["video", "p", "r", "f", "f_bigram", "f_rougeL", "combo",
                "gt_len", "pred_len", "len_ratio", "p_raw", "r_raw", "f_raw"]
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in sorted(rows, key=lambda x: x["f"]):
                w.writerow(r)
        print(f"\n📄 {args.csv} yazıldı ({len(rows)} satır).")

    print("\n" + "=" * 74 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())