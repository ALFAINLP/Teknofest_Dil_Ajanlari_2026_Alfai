"""
kpi_enhanced.py — Şartname uyumlu KPI ölçüm katmanı (Alfai)

TEKNOFEST 2026 TYDA 3. Senaryo, Madde 4 "Ölçümleme ve KPI Tanımlama"
kapsamında tanımlanan metrikler:

    Şartname kalemi              -> Bu moduldeki karşılığı
    ---------------------------------------------------------------
    Olay tespit doğruluğu        -> event_f1_score (aralık bazlı, IoM)
    Kritik olay yakalama oranı   -> critical_recall (ayrı katman)
    Özet kalitesi                -> summary_score (ROUGE-1/2 + özlülük)
    Aksiyon önerisi doğruluğu    -> tool_jaccard_iou + risk_accuracy
    İşlem süresi                 -> total_processing_time_sec, RTF
    Bellek ve donanım kullanımı  -> peak_gpu_gb, peak_rss_gb (örneklemeli)

Kullanım
--------
    python kpi_enhanced.py                 # benchmark
    python kpi_enhanced.py --self-test     # ölçüm çekirdeğini doğrula
    python kpi_enhanced.py --print-metrics # metrik tanımlarını yazdır
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from io import StringIO
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import logging

# ============================================================================
# 0. AYARLAR VE METRİK TANIMLARI
# ============================================================================

TOLERANCE_RATIO = 0.05
TOLERANCE_MIN_SEC = 2.0
TOLERANCE_MAX_SEC = 5.0

IOM_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.04
SCORE_W_TIME = 0.6
SCORE_W_TEXT = 0.4

STEM_LEN = 5

SUMMARY_W_UNIGRAM = 0.60 
SUMMARY_W_BIGRAM = 0.40  
SUMMARY_LENGTH_TOLERANCE = 1.8     # GT özetinin 1.8 katına kadar ceza yok
SUMMARY_MIN_PENALTY = 0.80

VALID_PHASES = {"baslangic", "gelisim", "sonuc"}
SYNTHETIC_EVENT_TYPES = {"EK ANALIZ", "EK_ANALIZ", "EK ANALİZ", "EKANALIZ"}

NON_EVENT_PATTERNS = (
    "olay bulunamadi", "olay yok", "olay tespit edilmedi", "tespit edilmedi",
    "bilinmeyen olay", "aciklama bulunamadi", "veri yok", "bulunamadi",
)

RESOURCE_SAMPLE_INTERVAL = 2.0

# Şartname Md.4'e göre güncellenmiş katman ağırlıkları (Toplam = 1.00)
LAYER_WEIGHTS = {
    "event": 0.20,       # Olay tespit doğruluğu
    "critical": 0.20,    # Kritik olay yakalama oranı
    "summary": 0.20,     # Özet kalitesi
    "tool": 0.20,        # Aksiyon önerisi doğruluğu
    "risk": 0.20,        # Risk değerlendirmesi
}

METRIC_DEFINITIONS = {
    "olay_tespit_dogrulugu": {
        "alan": "Olay Tespiti",
        "tanim": "Temporal IoM (Intersection over Minimum) ve metin benzerliği tabanlı aralık eşleştirmeli F1 Skoru.",
        "esikler": f"IoM >= {IOM_THRESHOLD}, Metin >= {TEXT_THRESHOLD}",
        "gerekce": "Olay süreçlerinin başlangıç-gelişim-sonuç aralıklarını yakalama başarısını ölçer."
    },
    "kritik_olay_yakalama_orani": {
        "alan": "Güvenlik / Risk",
        "tanim": "Ground Truth'ta 'is_critical' olarak etiketlenmiş olayların yakalanma oranı (Recall).",
        "gerekce": "İş güvenliği ihlali ve kaza anlarının kaçırılmamasını garanti eder."
    },
    "ozet_kalitesi": {
        "alan": "Doğal Dil Üretimi",
        "tanim": "ROUGE-1/2 F1 harmonik ortalaması ve uzunluk cezası katsayısı.",
        "gerekce": "Operatöre sunulan özetin gereksiz detaydan arındırılmış ve anlam bütünlüğüne sahip olmasını sağlar."
    },
    "aksiyon_onerisi_dogrulugu": {
        "alan": "Karar Destek / Tool Çağrısı",
        "tanim": "Öngörülen araçlar ile Ground Truth araçları arasındaki Jaccard IoU küme benzerliği.",
        "gerekce": "Doğru protokole ve güvenlik adımına başvurulduğunu doğrular."
    },
    "risk_dogrulugu": {
        "alan": "Risk Değerlendirmesi",
        "tanim": "Nihai risk seviyesinin (Düşük/Orta/Yüksek/Kritik) Ground Truth ile tam eşleşmesi.",
        "gerekce": "Saha riskinin doğru sınıflandırıldığını teyit eder."
    },
    "islem_suresi": {
        "alan": "Performans",
        "tanim": "Pipeline toplam çalışma süresi ve Real-Time Factor (RTF).",
        "gerekce": "Videonun gerçek zamanlı işlenme yeteneğini ölçer."
    },
    "bellek_ve_donanim": {
        "alan": "Kaynak Tüketimi",
        "tanim": "Çalışma anındaki maksimum GPU bellek tüketimi (peak VRAM GB) ve işlem RAM tüketimi (peak RSS GB).",
        "gerekce": "Uç cihaz ve sunucu kaynak sınırlarına uyumu takip eder."
    }
}

logging.getLogger("ALFAI_API").setLevel(logging.ERROR)
logging.getLogger("ALFAI_API").disabled = True
for _name in ("httpx", "openai", "vllm", "uvicorn", "fastapi"):
    logging.getLogger(_name).setLevel(logging.ERROR)
    logging.getLogger(_name).disabled = True
logging.basicConfig(level=logging.ERROR)


def _mean(values: Sequence[Optional[float]], default: float = 0.0) -> float:
    vals = [v for v in values if v is not None]
    return statistics.fmean(vals) if vals else default


# ============================================================================
# 1. TÜRKÇE METİN İŞLEME
# ============================================================================

class TurkishTextProcessor:
    _TR_MAP = {
        "ı": "i", "ğ": "g", "ü": "u", "ş": "s", "ö": "o", "ç": "c",
        "Ğ": "g", "Ü": "u", "Ş": "s", "Ö": "o", "Ç": "c",
    }

    TR_STOP_WORDS = {
        "ve", "ile", "icin", "bir", "bu", "su", "da", "de", "ki",
        "mi", "mu", "ne", "nasil", "neden", "niye", "cunku",
        "ama", "fakat", "ancak", "lakin", "yani", "demek", "soyle", "boyle",
        "oyle", "simdi", "sonra", "once", "tarafindan", "olarak",
        "girdi", "cikti", "iken", "gibi", "kadar", "daha", "cok", "her",
        "olan", "olup", "ise", "ancak", "ayrica", "yine", "hem",
        "description", "type", "event", "time", "zone", "risk", "confidence",
    }

    @classmethod
    def normalize(cls, text: Any) -> str:
        text = re.sub(r"[İI]", "i", str(text or ""))
        for tr, en in cls._TR_MAP.items():
            text = text.replace(tr, en)
        return text.lower().strip()

    @classmethod
    def tokens(cls, text: Any, min_len: int = 3) -> List[str]:
        words = re.findall(r"[a-z]{2,}", cls.normalize(text))
        return [w for w in words if w not in cls.TR_STOP_WORDS and len(w) >= min_len]

    @classmethod
    def stem_tokens(cls, text: Any) -> List[str]:
        return [w[:STEM_LEN] for w in cls.tokens(text) if len(w) >= STEM_LEN]

    @classmethod
    def keywords(cls, text: Any, min_len: int = 3) -> Set[str]:
        return set(cls.tokens(text, min_len))

    @classmethod
    def stems(cls, text: Any) -> Set[str]:
        return set(cls.stem_tokens(text))

    @staticmethod
    def _jaccard(a: Set[str], b: Set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    @classmethod
    def similarity(cls, text_a: Any, text_b: Any) -> float:
        raw = cls._jaccard(cls.keywords(text_a), cls.keywords(text_b))
        stem = cls._jaccard(cls.stems(text_a), cls.stems(text_b))
        return max(raw, stem)


# ============================================================================
# 2. ÖZET KALİTESİ
# ============================================================================

def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(pred_tokens: Sequence[str], gt_tokens: Sequence[str], n: int) -> Dict[str, float]:
    p, g = _ngram_counts(pred_tokens, n), _ngram_counts(gt_tokens, n)
    p_total, g_total = sum(p.values()), sum(g.values())
    if p_total == 0 or g_total == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    overlap = sum((p & g).values())
    precision = overlap / p_total
    recall = overlap / g_total
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


class SummaryEvaluator:
    @staticmethod
    def evaluate(pred_summary: Any, gt_summary: Any) -> Dict[str, Any]:
        gt_text = str(gt_summary or "").strip()
        pred_text = str(pred_summary or "").strip()

        if not gt_text:
            return {"summary_score": None, "reason": "GT özeti yok, ölçülemedi"}
        if not pred_text:
            return {
                "summary_score": 0.0, "rouge1_f1": 0.0, "rouge2_f1": 0.0,
                "rouge1_recall": 0.0, "length_ratio": 0.0,
                "conciseness_penalty": 1.0, "reason": "Model özet üretmedi",
            }

        pred_tok = TurkishTextProcessor.stem_tokens(pred_text)
        gt_tok = TurkishTextProcessor.stem_tokens(gt_text)

        if not gt_tok:
            return {"summary_score": None, "reason": "GT özetinde anlamlı kelime yok"}

        r1 = rouge_n(pred_tok, gt_tok, 1)
        r2 = rouge_n(pred_tok, gt_tok, 2)

        length_ratio = len(pred_tok) / len(gt_tok)
        if length_ratio <= SUMMARY_LENGTH_TOLERANCE:
            penalty = 1.0
        else:
            penalty = max(SUMMARY_MIN_PENALTY, SUMMARY_LENGTH_TOLERANCE / length_ratio)
        base = SUMMARY_W_UNIGRAM * r1["f1"] + SUMMARY_W_BIGRAM * r2["f1"]
        score = base * penalty

        return {
            "summary_score": round(score, 3),
            "rouge1_f1": round(r1["f1"], 3),
            "rouge1_recall": round(r1["recall"], 3),
            "rouge1_precision": round(r1["precision"], 3),
            "rouge2_f1": round(r2["f1"], 3),
            "length_ratio": round(length_ratio, 2),
            "conciseness_penalty": round(penalty, 3),
            "pred_word_count": len(pred_tok),
            "gt_word_count": len(gt_tok),
        }


# ============================================================================
# 3. KAYNAK İZLEME
# ============================================================================

class ResourceMonitor:
    def __init__(self, interval: float = RESOURCE_SAMPLE_INTERVAL):
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.gpu_samples: List[float] = []
        self.rss_samples: List[float] = []
        self.baseline_gpu: Optional[float] = None
        self.gpu_available = False

    @staticmethod
    def _read_gpu_gb() -> Optional[float]:
        try:
            import pynvml  # type: ignore
            if not getattr(ResourceMonitor, "_nvml_ready", False):
                pynvml.nvmlInit()
                ResourceMonitor._nvml_ready = True
            total = 0.0
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                total += pynvml.nvmlDeviceGetMemoryInfo(h).used
            return total / (1024 ** 3)
        except Exception:
            pass
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
                encoding="utf-8", timeout=5, stderr=subprocess.DEVNULL,
            )
            return sum(float(x) for x in out.strip().splitlines() if x.strip()) / 1024
        except Exception:
            return None

    @staticmethod
    def _read_rss_gb() -> Optional[float]:
        try:
            import psutil  # type: ignore
            return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
        except Exception:
            try:
                with open(f"/proc/{os.getpid()}/status", "r") as fh:
                    for line in fh:
                        if line.startswith("VmRSS:"):
                            return float(line.split()[1]) / (1024 ** 2)
            except Exception:
                pass
            return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            gpu = self._read_gpu_gb()
            if gpu is not None:
                self.gpu_samples.append(gpu)
            rss = self._read_rss_gb()
            if rss is not None:
                self.rss_samples.append(rss)
            self._stop.wait(self.interval)

    def __enter__(self) -> "ResourceMonitor":
        self.baseline_gpu = self._read_gpu_gb()
        self.gpu_available = self.baseline_gpu is not None
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2 * self.interval + 1)

    def report(self) -> Dict[str, Any]:
        peak_gpu = max(self.gpu_samples) if self.gpu_samples else None
        delta = None
        if peak_gpu is not None and self.baseline_gpu is not None:
            delta = max(0.0, peak_gpu - self.baseline_gpu)
        return {
            "peak_gpu_gb": round(peak_gpu, 3) if peak_gpu is not None else None,
            "baseline_gpu_gb": round(self.baseline_gpu, 3) if self.baseline_gpu is not None else None,
            "gpu_delta_gb": round(delta, 3) if delta is not None else None,
            "peak_rss_gb": round(max(self.rss_samples), 3) if self.rss_samples else None,
            "samples": len(self.gpu_samples),
            "note": None if self.gpu_available else "nvidia-smi okunamadı",
        }


# ============================================================================
# 4. ZAMAN AYRIŞTIRMA
# ============================================================================

def parse_time(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return v if v >= 0 else None
    if not isinstance(value, str):
        return None

    s = value.strip()
    if not s:
        return None
    s = re.sub(r"\b(sn|sec|saniye|s)\b", "", s, flags=re.IGNORECASE).strip()

    if ":" in s:
        try:
            nums = [float(p.replace(",", ".")) for p in s.split(":")]
        except ValueError:
            return None
        if len(nums) == 2:
            total = nums[0] * 60 + nums[1]
        elif len(nums) == 3:
            total = nums[0] * 3600 + nums[1] * 60 + nums[2]
        else:
            return None
        return total if total >= 0 else None

    try:
        v = float(s.replace(",", "."))
    except ValueError:
        return None
    return v if v >= 0 else None


def parse_range(value: Any) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(value, str) or not re.search(r"\d\s*[-–—]\s*\d", value):
        return None, None
    parts = re.split(r"\s*[-–—]\s*", value.strip(), maxsplit=1)
    if len(parts) != 2:
        return None, None
    return parse_time(parts[0]), parse_time(parts[1])


@dataclass(frozen=True)
class Interval:
    start: float
    end: float

    @property
    def length(self) -> float:
        return max(0.0, self.end - self.start)

    def overlap(self, other: "Interval") -> float:
        return max(0.0, min(self.end, other.end) - max(self.start, other.start))

    def iom(self, other: "Interval") -> float:
        denom = min(self.length, other.length)
        return self.overlap(other) / denom if denom > 0 else 0.0


def tolerance_for_duration(duration_sec: Optional[float]) -> float:
    if not duration_sec or duration_sec <= 0:
        return TOLERANCE_MIN_SEC
    return max(TOLERANCE_MIN_SEC, min(TOLERANCE_MAX_SEC, TOLERANCE_RATIO * float(duration_sec)))


def event_interval(event: Any, pad: float,
                   duration_sec: Optional[float] = None,
                   is_ground_truth: bool = False) -> Optional[Interval]:
    if not isinstance(event, dict):
        return None

    start = parse_time(event.get("start_time"))
    end = parse_time(event.get("end_time"))

    if start is None and end is None:
        rs, re_ = parse_range(event.get("time"))
        if rs is not None:
            start, end = rs, re_

    if start is not None and end is not None and end > start:
        lo, hi = start, end
    else:
        anchor = start if start is not None else parse_time(event.get("time"))
        if anchor is None:
            anchor = end
        if anchor is None:
            return None
        lo, hi = anchor - pad, anchor + pad

    lo = max(0.0, lo)
    if duration_sec and duration_sec > 0 and not is_ground_truth:
        hi = min(hi, float(duration_sec))
    if hi <= lo:
        hi = lo + max(0.5, pad)
    return Interval(lo, hi)


# ============================================================================
# 5. OLAY DEĞERLENDİRİCİ
# ============================================================================

@dataclass
class MatchPair:
    gt_index: int
    pred_index: int
    score: float
    iom: float
    text_sim: float


class TemporalEventEvaluator:
    def __init__(self, tolerance_sec: float = TOLERANCE_MIN_SEC):
        self.tolerance = float(tolerance_sec)
        self.tr = TurkishTextProcessor()

    def set_tolerance_from_duration(self, duration_sec: Optional[float]) -> float:
        self.tolerance = tolerance_for_duration(duration_sec)
        return self.tolerance

    @staticmethod
    def _event_text(event: Any) -> str:
        if not isinstance(event, dict):
            return str(event or "")
        raw = event.get("event", event.get("description", ""))
        if isinstance(raw, dict):
            raw = raw.get("description") or raw.get("event") or " ".join(
                str(v) for v in raw.values() if isinstance(v, str))
        return str(raw or "")

    @staticmethod
    def _is_critical(event: Any) -> bool:
        if not isinstance(event, dict):
            return False
        return bool(event.get("is_critical") or event.get("is_critical_moment"))

    @staticmethod
    def _is_synthetic(event: Any) -> bool:
        if not isinstance(event, dict):
            return False
        return str(event.get("event_type", "")).strip().upper() in SYNTHETIC_EVENT_TYPES

    @classmethod
    def _is_non_event(cls, event: Any) -> bool:
        text = TurkishTextProcessor.normalize(cls._event_text(event))
        if len(text) < 3:
            return True
        return any(p in text for p in NON_EVENT_PATTERNS)

    @staticmethod
    def _phase_diagnostics(pred_events: List[Dict]) -> Dict[str, Any]:
        counts: Counter = Counter()
        for e in pred_events:
            if not isinstance(e, dict):
                continue
            phase = TurkishTextProcessor.normalize(e.get("phase", ""))
            counts[phase if phase in VALID_PHASES else "tanimsiz"] += 1
        total = sum(counts.values())
        distinct = len([p for p in VALID_PHASES if counts.get(p, 0) > 0])
        return {
            "distribution": dict(counts),
            "labelled_ratio": round((total - counts.get("tanimsiz", 0)) / total, 3) if total else None,
            "distinct_phases": distinct,
            "note": "GT'de phase alanı yok; yalnızca üretim kapsamı",
        }

    @staticmethod
    def _assign(pairs: List[MatchPair], n_gt: int, n_pred: int) -> List[MatchPair]:
        if not pairs:
            return []
        try:
            from scipy.optimize import linear_sum_assignment  # type: ignore
            BIG = 1e6
            cost = [[BIG] * n_pred for _ in range(n_gt)]
            lookup: Dict[Tuple[int, int], MatchPair] = {}
            for p in pairs:
                cost[p.gt_index][p.pred_index] = -p.score
                lookup[(p.gt_index, p.pred_index)] = p
            rows, cols = linear_sum_assignment(cost)
            out = [lookup[(int(r), int(c))] for r, c in zip(rows, cols)
                   if (int(r), int(c)) in lookup]
            return sorted(out, key=lambda p: p.gt_index)
        except Exception:
            pass

        used_gt: Set[int] = set()
        used_pred: Set[int] = set()
        out: List[MatchPair] = []
        for p in sorted(pairs, key=lambda x: (-x.score, x.gt_index, x.pred_index)):
            if p.gt_index in used_gt or p.pred_index in used_pred:
                continue
            used_gt.add(p.gt_index)
            used_pred.add(p.pred_index)
            out.append(p)
        return sorted(out, key=lambda p: p.gt_index)

    def evaluate(self, pred_events: List[Dict], true_events: List[Dict],
                 duration_sec: Optional[float] = None,
                 drop_synthetic: bool = False) -> Dict[str, Any]:
        pred_events = list(pred_events or [])
        true_events = list(true_events or [])

        dropped = 0
        if drop_synthetic:
            kept = [e for e in pred_events if not self._is_synthetic(e)]
            dropped = len(pred_events) - len(kept)
            pred_events = kept

        before = len(pred_events)
        pred_events = [e for e in pred_events if not self._is_non_event(e)]
        non_events = before - len(pred_events)

        phase_diag = self._phase_diagnostics(pred_events)
        n_gt, n_pred = len(true_events), len(pred_events)

        if n_gt == 0 and n_pred == 0:
            return self._empty(1.0, n_gt, n_pred, dropped, phase_diag, non_events)
        if n_gt == 0:
            res = self._empty(0.0, n_gt, n_pred, dropped, phase_diag, non_events)
            res.update({"fp": n_pred})
            return res
        if n_pred == 0:
            res = self._empty(0.0, n_gt, n_pred, dropped, phase_diag, non_events)
            crit = sum(1 for e in true_events if self._is_critical(e))
            res.update({"fn": n_gt, "critical_gt_count": crit,
                        "critical_recall": 0.0 if crit else None})
            return res

        pad = self.tolerance
        gt_iv = [event_interval(e, pad, duration_sec, is_ground_truth=True) for e in true_events]
        pred_iv = [event_interval(e, pad, duration_sec) for e in pred_events]
        gt_txt = [self._event_text(e) for e in true_events]
        pred_txt = [self._event_text(e) for e in pred_events]

        candidates: List[MatchPair] = []
        for gi in range(n_gt):
            if gt_iv[gi] is None:
                continue
            for pi in range(n_pred):
                if pred_iv[pi] is None:
                    continue
                iom = gt_iv[gi].iom(pred_iv[pi])
                if iom < IOM_THRESHOLD:
                    continue
                sim = self.tr.similarity(gt_txt[gi], pred_txt[pi])
                if sim < TEXT_THRESHOLD:
                    continue
                candidates.append(
                    MatchPair(gi, pi, SCORE_W_TIME * iom + SCORE_W_TEXT * sim, iom, sim))

        matches = self._assign(candidates, n_gt, n_pred)
        matched_gt = {m.gt_index for m in matches}
        matched_pred = {m.pred_index for m in matches}

        tp = len(matches)
        fp = n_pred - len(matched_pred)
        fn = n_gt - len(matched_gt)
        assert tp + fp == n_pred, f"tp+fp={tp+fp} != len(pred)={n_pred}"
        assert tp + fn == n_gt, f"tp+fn={tp+fn} != len(gt)={n_gt}"

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        crit_gt = [i for i in range(n_gt) if self._is_critical(true_events[i])]
        crit_hit = [i for i in crit_gt if i in matched_gt]
        critical_recall = (len(crit_hit) / len(crit_gt)) if crit_gt else None

        flag_ok = sum(1 for m in matches
                      if self._is_critical(true_events[m.gt_index])
                      == self._is_critical(pred_events[m.pred_index]))
        flag_acc = (flag_ok / tp) if tp else None

        mean_iom = _mean([m.iom for m in matches])
        mean_text = _mean([m.text_sim for m in matches])
        weighted = 0.70 * f1 + 0.15 * mean_iom + 0.15 * mean_text

        return {
            "f1_score": round(f1, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "weighted_final_score": round(weighted, 3),
            "tp": tp, "fp": fp, "fn": fn,
            "total_gt_events": n_gt, "detected_events": n_pred,
            "tolerance_sec": round(pad, 2),
            "mean_iom": round(mean_iom, 3),
            "mean_text_similarity": round(mean_text, 3),
            "critical_recall": round(critical_recall, 3) if critical_recall is not None else None,
            "critical_gt_count": len(crit_gt),
            "critical_matched": len(crit_hit),
            "critical_flag_accuracy": round(flag_acc, 3) if flag_acc is not None else None,
            "unparsable_pred_times": sum(1 for iv in pred_iv if iv is None),
            "unparsable_gt_times": sum(1 for iv in gt_iv if iv is None),
            "dropped_synthetic_events": dropped,
            "non_event_predictions": non_events,
            "phase_diagnostics": phase_diag,
            "matches": [
                {"gt": gt_txt[m.gt_index][:70],
                 "gt_interval": [round(gt_iv[m.gt_index].start, 1), round(gt_iv[m.gt_index].end, 1)],
                 "pred": pred_txt[m.pred_index][:70],
                 "pred_interval": [round(pred_iv[m.pred_index].start, 1), round(pred_iv[m.pred_index].end, 1)],
                 "iom": round(m.iom, 3), "text_sim": round(m.text_sim, 3)}
                for m in matches
            ],
            "missed_gt": [
                {"text": gt_txt[i][:70],
                 "critical": self._is_critical(true_events[i]),
                 "interval": [round(gt_iv[i].start, 1), round(gt_iv[i].end, 1)] if gt_iv[i] else None}
                for i in range(n_gt) if i not in matched_gt
            ],
            "false_positives": [
                {"text": pred_txt[i][:70],
                 "interval": [round(pred_iv[i].start, 1), round(pred_iv[i].end, 1)] if pred_iv[i] else None}
                for i in range(n_pred) if i not in matched_pred
            ],
        }

    @staticmethod
    def _empty(score: float, n_gt: int, n_pred: int, dropped: int,
               phase_diag: Dict[str, Any], non_events: int = 0) -> Dict[str, Any]:
        return {
            "f1_score": score, "precision": score, "recall": score,
            "weighted_final_score": score,
            "tp": 0, "fp": 0, "fn": 0,
            "total_gt_events": n_gt, "detected_events": n_pred,
            "tolerance_sec": None, "mean_iom": 0.0, "mean_text_similarity": 0.0,
            "critical_recall": None, "critical_gt_count": 0, "critical_matched": 0,
            "critical_flag_accuracy": None,
            "unparsable_pred_times": 0, "unparsable_gt_times": 0,
            "dropped_synthetic_events": dropped,
            "non_event_predictions": non_events,
            "phase_diagnostics": phase_diag,
            "matches": [], "missed_gt": [], "false_positives": [],
        }


# ============================================================================
# 6. TOOL DEĞERLENDİRİCİ
# ============================================================================

class ToolEvaluator:
    TOOL_ALIASES = {
        "trigger_operator_alert": ["operator_alert", "trigger_alert"],
        "get_personnel_on_duty": ["get_personnel", "personnel_on_duty"],
        "assess_environmental_hazards": ["environmental_hazards", "assess_hazards"],
        "query_previous_incidents": ["previous_incidents", "query_incident"],
        "export_to_incident_management_system": ["export_incident", "incident_management"],
        "detect_motion_density": ["motion_density"],
        "log_kpi_metrics": ["kpi_metrics", "log_kpi"],
    }

    def __init__(self):
        self.alias_to_tool: Dict[str, str] = {}
        for tool, aliases in self.TOOL_ALIASES.items():
            self.alias_to_tool[tool] = tool
            for alias in aliases:
                self.alias_to_tool[alias] = tool

    def extract(self, actions: Sequence[Any]) -> Set[str]:
        found: Set[str] = set()
        for action in actions or []:
            norm = TurkishTextProcessor.normalize(action)
            if norm in self.alias_to_tool:
                found.add(self.alias_to_tool[norm])
                continue
            for alias, tool in self.alias_to_tool.items():
                if re.search(rf"(?<![a-z0-9_]){re.escape(alias)}(?![a-z0-9_])", norm):
                    found.add(tool)
        return found

    def evaluate(self, pred_actions: Sequence[Any],
                 true_actions: Sequence[Any]) -> Dict[str, Any]:
        pred = self.extract(pred_actions)
        true = self.extract(true_actions)

        if not true and not pred:
            return {
                "tool_jaccard_iou": None,
                "tool_recall": None,
                "tool_precision": None,
                "hallucination_rate": None,
                "weighted_final_score": None,
                "detected_tools": [],
                "expected_tools": [],
                "tp": 0, "fp": 0, "fn": 0,
                "evaluated": False,
                "reason": "Beklenen veya tespit edilen tool yok; tool katmanı ölçülmedi"
            }

        if not true and pred:
            return {
                "tool_jaccard_iou": 0.0,
                "tool_recall": 0.0,
                "tool_precision": 0.0,
                "hallucination_rate": 1.0,
                "weighted_final_score": 0.0,
                "detected_tools": sorted(pred),
                "expected_tools": [],
                "tp": 0, "fp": len(pred), "fn": 0,
                "evaluated": True,
                "reason": "GT'de tool beklenmiyordu ancak model tool çağırdı"
            }

        tp = len(pred & true)
        fp = len(pred - true)
        fn = len(true - pred)

        denom = tp + fp + fn
        iou = tp / denom if denom else 0.0
        recall = tp / len(true) if true else 0.0
        precision = tp / len(pred) if pred else 0.0
        hallucination_rate = (fp / len(pred)) if pred else 0.0

        return {
            "tool_jaccard_iou": round(iou, 3),
            "tool_recall": round(recall, 3),
            "tool_precision": round(precision, 3),
            "hallucination_rate": round(hallucination_rate, 3),
            "weighted_final_score": round(iou, 3),
            "tool_exact_match": 1.0 if pred == true else 0.0,
            "detected_tools": sorted(pred),
            "expected_tools": sorted(true),
            "tp": tp, "fp": fp, "fn": fn,
            "evaluated": True,
            "reason": "Tool çağrısı beklenen senaryo ölçüldü"
        }


# ============================================================================
# 7. ANA BENCHMARK
# ============================================================================

class EnhancedAgentBenchmarkKPI:
    def __init__(self, drop_synthetic_events: bool = False):
        self.tr = TurkishTextProcessor()
        self.tool_evaluator = ToolEvaluator()
        self.event_evaluator = TemporalEventEvaluator()
        self.summary_evaluator = SummaryEvaluator()
        self.drop_synthetic_events = drop_synthetic_events

    def evaluate_risk(self, pred: Dict, true: Dict) -> Optional[float]:
        p = self.tr.normalize(pred.get("risk_level", pred.get("risk", "")))
        t = self.tr.normalize(true.get("risk_level", true.get("risk", "")))
        if not p or not t:
            return None
        return 1.0 if p == t else 0.0

    @staticmethod
    def _is_tool_error(step: Dict) -> bool:
        if not isinstance(step, dict):
            return False
        if step.get("error") or step.get("status") == "error":
            return True
        res = step.get("result")
        if isinstance(res, dict) and "error" in res:
            return True
        return False

    def evaluate_recovery(self, full_state: Dict) -> Dict[str, Any]:
        """Hata toparlanma değerlendirmesi:
        ERROR -> Retry / alternatif Tool -> Başarılı sonuç?
          ├── EVET -> Recovery = 1.0
          └── HAYIR -> Recovery = 0.0
        """
        history = [t for t in full_state.get("tool_history", []) if isinstance(t, dict)]
        if not history:
            return {
                "recovery_score": None,
                "evaluated": False,
                "reason": "Tool geçmişi boş; recovery ölçülmedi"
            }

        # Hatanın oluştuğu ilk adımı bul
        first_error_idx = None
        for i, step in enumerate(history):
            if self._is_tool_error(step):
                first_error_idx = i
                break

        if first_error_idx is None:
            return {
                "recovery_score": None,
                "evaluated": False,
                "reason": "Hata oluşmadı; recovery ölçülmedi"
            }

        subsequent_calls = history[first_error_idx + 1:]
        if not subsequent_calls:
            return {
                "recovery_score": 0.0,
                "evaluated": True,
                "recovered": False,
                "reason": "Hata sonrası tekrar deneme / alternatif araç çağrılmadı"
            }

        # Sonraki adımda başarılı sonuç alındı mı?
        recovered = False
        for next_step in subsequent_calls:
            if not self._is_tool_error(next_step):
                recovered = True
                break

        recovery_score = 1.0 if recovered else 0.0
        return {
            "recovery_score": recovery_score,
            "evaluated": True,
            "recovered": recovered,
            "reason": "Hata sonrası başarılı toparlanma sağlandı" if recovered else "Hata toparlanamadı"
        }

    @staticmethod
    def _overall(scores: Dict[str, Optional[float]]) -> Tuple[float, List[str]]:
        available = [k for k in LAYER_WEIGHTS if scores.get(k) is not None]
        total_w = sum(LAYER_WEIGHTS[k] for k in available)
        if total_w <= 0:
            return 0.0, []
        acc = sum(scores[k] * LAYER_WEIGHTS[k] for k in available)
        skipped = [k for k in LAYER_WEIGHTS if k not in available]
        return acc / total_w, skipped

    def run_benchmark(self, pipeline_fn, video_path: str, ground_truth: Dict[str, Any],
                      video_duration_sec: float) -> Dict[str, Any]:
        tolerance = self.event_evaluator.set_tolerance_from_duration(video_duration_sec)

        with ResourceMonitor() as monitor:
            start = time.perf_counter()
            prediction = pipeline_fn(video_path)
            latency = time.perf_counter() - start
        resources = monitor.report()

        full_state = prediction.get("_full_state", {})
        rtf = latency / video_duration_sec if video_duration_sec > 0 else None

        pred_summary_text = prediction.get("summary", "")
        gt_summary_text = ground_truth.get("summary", "")

        event_res = self.event_evaluator.evaluate(
            prediction.get("events", []), ground_truth.get("events", []),
            duration_sec=video_duration_sec, drop_synthetic=self.drop_synthetic_events)
        summary_res = self.summary_evaluator.evaluate(
            pred_summary_text, gt_summary_text)
        tool_res = self.tool_evaluator.evaluate(
            prediction.get("actions", []), ground_truth.get("actions", []))
        risk_score = self.evaluate_risk(prediction, ground_truth)
        recovery_res = self.evaluate_recovery(full_state)

        layer_scores = {
            "event": event_res["weighted_final_score"],
            "critical": event_res["critical_recall"],
            "summary": summary_res["summary_score"],
            "tool": tool_res["weighted_final_score"],
            "risk": risk_score,
        }
        overall, skipped = self._overall(layer_scores)

        return {
            "performance": {
                "total_processing_time_sec": round(latency, 3),
                "real_time_factor": round(rtf, 3) if rtf is not None else None,
                "peak_gpu_gb": resources["peak_gpu_gb"],
                "gpu_delta_gb": resources["gpu_delta_gb"],
                "peak_rss_gb": resources["peak_rss_gb"],
                "resource_samples": resources["samples"],
                "resource_note": resources["note"],
                "duration_sec_used": video_duration_sec,
            },
            "accuracy": {
                "event_f1_score": event_res["f1_score"],
                "event_precision": event_res["precision"],
                "event_recall": event_res["recall"],
                "critical_recall": event_res["critical_recall"],
                "summary_score": summary_res["summary_score"],
                "summary_text": pred_summary_text,
                "gt_summary_text": gt_summary_text,
                "risk_accuracy": risk_score,
                "tool_jaccard_score": tool_res["tool_jaccard_iou"],
                "events": prediction.get("events", []),
            },
            "layer_analysis": {
                "event_analysis": event_res,
                "summary_analysis": summary_res,
                "tool_selection": tool_res,
                "error_recovery": recovery_res,
                "layer_scores": layer_scores,
                "layer_weights": LAYER_WEIGHTS,
                "skipped_layers": skipped,
                "tolerance_sec": round(tolerance, 2),
                "overall_score": round(overall, 3),
            },
            "prediction_schema_valid": prediction.get("_schema_valid"),
        }


# ============================================================================
# 8. PIPELINE + VİDEO SÜRESİ
# ============================================================================

def get_video_duration(video_path: str, fallback: Optional[float] = None) -> Optional[float]:
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if fps and fps > 0 and frames and frames > 0:
                return round(frames / fps, 2)
        cap.release()
    except Exception:
        pass
    return fallback


def run_agent_pipeline(video_path: str) -> dict:
    from agent.graph import app  # lazy import

    initial_state = {
        "video_path": video_path,
        "current_time_segments": [], "detected_events": [],
        "risk_assessments": [], "actions_recommended": [],
        "final_json_output": "", "next_step": "",
        "event_history": [], "risk_history": [],
        "tool_history": [], "messages": [],
    }

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        final_output = app.invoke(initial_state)
    finally:
        sys.stdout = old_stdout

    final_json = final_output.get("final_json_output", "")
    parsed: Dict[str, Any] = {}
    if isinstance(final_json, str) and final_json.strip():
        try:
            cleaned = re.sub(r"^```json\s*|\s*```$", "", final_json.strip(), flags=re.MULTILINE)
            parsed = json.loads(cleaned)
        except Exception:
            parsed = {}
    elif isinstance(final_json, dict):
        parsed = final_json

    schema_valid: Optional[bool] = None
    if parsed:
        try:
            from schemas.output_schema import FinalReportSchema
            FinalReportSchema(**parsed)
            schema_valid = True
        except Exception:
            schema_valid = False

    risk_list = final_output.get("risk_assessments") or []
    parsed_risk = parsed.get("risk", parsed.get("risk_level", ""))
    risk_level = parsed_risk or (risk_list[0] if risk_list else "")
    return {
        "risk_level": risk_level,
        "summary": parsed.get("summary", final_output.get("video_summary", "")),
        "events": parsed.get("events", final_output.get("detected_events", [])),
        "actions": list(set(final_output.get("actions_recommended", []))),
        "_schema_valid": schema_valid,
        "_full_state": final_output,
    }


# ============================================================================
# 9. RAPORLAMA
# ============================================================================

def run_enhanced_benchmark(ground_truth_path: str = "ground_truth.json",
                           videos_dir: str = "Videos",
                           output_path: str = "benchmark_results_enhanced.json",
                           drop_synthetic_events: bool = False) -> None:
    if not os.path.exists(ground_truth_path):
        print(f"❌ Hata: '{ground_truth_path}' bulunamadı.")
        return

    with open(ground_truth_path, "r", encoding="utf-8") as f:
        gt_list = json.load(f)

    bench = EnhancedAgentBenchmarkKPI(drop_synthetic_events=drop_synthetic_events)
    results: List[Dict[str, Any]] = []
    failures: List[str] = []
    duration_warnings: List[str] = []
    schema_failures: List[str] = []

    m: Dict[str, List[float]] = {k: [] for k in (
        "event_f1", "event_precision", "event_recall", "critical_recall",
        "summary", "risk", "tool_jaccard", "overall",
        "latency", "rtf", "peak_gpu", "peak_rss")}

    print(f"\n🚀 {len(gt_list)} video için KPI Benchmark başlatılıyor...")
    print("=" * 66)

    for idx, item in enumerate(gt_list, 1):
        video_id = item.get("video_id", "")
        video_path = os.path.join(videos_dir, video_id)
        gt_duration = float(item.get("duration_sec", 0) or 0)

        if not os.path.exists(video_path):
            print(f"⚠️  Video bulunamadı: {video_path} (atlanıyor)")
            failures.append(f"{video_id}: dosya yok")
            continue

        duration = get_video_duration(video_path, fallback=gt_duration) or gt_duration
        if gt_duration and duration and abs(duration - gt_duration) > max(2.0, 0.1 * gt_duration):
            duration_warnings.append(f"{video_id}: GT={gt_duration:.0f}sn, gerçek={duration:.0f}sn")

        tol = tolerance_for_duration(duration)
        print(f"▶️  [{idx}/{len(gt_list)}] {video_id} ({duration:.0f} sn, ±{tol:.1f} sn) ...",
              end=" ", flush=True)

        try:
            res = bench.run_benchmark(run_agent_pipeline, video_path, item, duration)
        except Exception as exc:
            print(f"❌ {type(exc).__name__}: {str(exc)[:55]}")
            failures.append(f"{video_id}: {type(exc).__name__}: {exc}")
            continue

        summary_detail = {
            "ground_truth_summary": item.get("summary", ""),
            "model_summary": res.get("accuracy", {}).get("summary_text", ""),
            "summary_score": res.get("accuracy", {}).get("summary_score", 0.0),
            "summary_analysis": res.get("layer_analysis", {}).get("summary_analysis", {})
        }
        
        critical_events_detail = []
        gt_events = item.get("events", [])
        pred_events = res.get("accuracy", {}).get("events", [])
        
        def get_time_value(event, key):
            val = event.get(key)
            if val is None:
                return None
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                return parse_time(val)
            return None
        
        for gt_event in gt_events:
            if gt_event.get("is_critical", False):
                gt_start = get_time_value(gt_event, "start_time")
                gt_time = get_time_value(gt_event, "time")
                
                model_pred = None
                for p in pred_events:
                    p_start = get_time_value(p, "start_time")
                    p_time = get_time_value(p, "time")
                    
                    if p.get("time") == gt_event.get("time"):
                        model_pred = p
                        break
                    elif (p_start is not None and gt_start is not None and abs(p_start - gt_start) < 2.0):
                        model_pred = p
                        break
                    elif (p_time is not None and gt_time is not None and abs(p_time - gt_time) < 2.0):
                        model_pred = p
                        break
                
                is_matched = False
                for p in pred_events:
                    p_start = get_time_value(p, "start_time")
                    p_time = get_time_value(p, "time")
                    if p.get("time") == gt_event.get("time"):
                        is_matched = True
                        break
                    elif (p_start is not None and gt_start is not None and abs(p_start - gt_start) < 2.0):
                        is_matched = True
                        break
                    elif (p_time is not None and gt_time is not None and abs(p_time - gt_time) < 2.0):
                        is_matched = True
                        break
                
                critical_events_detail.append({
                    "time": gt_event.get("time", ""),
                    "start_time": gt_event.get("start_time", ""),
                    "end_time": gt_event.get("end_time", ""),
                    "event": gt_event.get("event", ""),
                    "phase": gt_event.get("phase", ""),
                    "ground_truth_summary": gt_event.get("detailed_description", item.get("summary", "")),
                    "model_prediction": model_pred if model_pred else {},
                    "is_matched": is_matched
                })
        
        time_series_summary = []
        for gt_event in gt_events:
            gt_start = get_time_value(gt_event, "start_time")
            gt_time = get_time_value(gt_event, "time")
            
            model_text = "Model bu olayı tespit edemedi"
            for p in pred_events:
                p_start = get_time_value(p, "start_time")
                p_time = get_time_value(p, "time")
                
                if p.get("time") == gt_event.get("time"):
                    model_text = p.get("event", "")
                    break
                elif (p_start is not None and gt_start is not None and abs(p_start - gt_start) < 2.0):
                    model_text = p.get("event", "")
                    break
                elif (p_time is not None and gt_time is not None and abs(p_time - gt_time) < 2.0):
                    model_text = p.get("event", "")
                    break
            
            time_info = {
                "time_range": f"{gt_event.get('start_time', gt_event.get('time', ''))} - {gt_event.get('end_time', gt_event.get('time', ''))}",
                "event": gt_event.get("event", ""),
                "is_critical": gt_event.get("is_critical", False),
                "ground_truth_text": gt_event.get("detailed_description", ""),
                "model_summary_text": model_text
            }
            time_series_summary.append(time_info)

        results.append({
            "video_id": video_id,
            "metrics": res,
            "summary_details": summary_detail,
            "critical_events": critical_events_detail,
            "time_series_summary": time_series_summary
        })
        
        acc, perf, layer = res["accuracy"], res["performance"], res["layer_analysis"]
        if res["prediction_schema_valid"] is False:
            schema_failures.append(video_id)
        print(f"✅ F1={acc['event_f1_score']:.2f} özet={acc['summary_score'] or 0:.2f} "
              f"genel={layer['overall_score']:.2f}")

        m["event_f1"].append(acc["event_f1_score"])
        m["event_precision"].append(acc["event_precision"])
        m["event_recall"].append(acc["event_recall"])
        if acc["critical_recall"] is not None:
            m["critical_recall"].append(acc["critical_recall"])
        if acc["summary_score"] is not None:
            m["summary"].append(acc["summary_score"])
        if acc["risk_accuracy"] is not None:
            m["risk"].append(acc["risk_accuracy"])
        if acc["tool_jaccard_score"] is not None:
            m["tool_jaccard"].append(acc["tool_jaccard_score"])

        m["overall"].append(layer["overall_score"])
        m["latency"].append(perf["total_processing_time_sec"])
        if perf["real_time_factor"] is not None:
            m["rtf"].append(perf["real_time_factor"])
        if perf["peak_gpu_gb"] is not None:
            m["peak_gpu"].append(perf["peak_gpu_gb"])
        if perf["peak_rss_gb"] is not None:
            m["peak_rss"].append(perf["peak_rss_gb"])

    if not results:
        print("❌ Hiçbir video işlenemedi.")
        for f in failures[:10]:
            print(f"   - {f}")
        return

    payload = {
        "metric_definitions": METRIC_DEFINITIONS,
        "layer_weights": LAYER_WEIGHTS,
        "run_summary": {
            "videos_total": len(gt_list),
            "videos_evaluated": len(results),
            "videos_failed": len(failures),
            "failures": failures,
            "duration_warnings": duration_warnings,
            "schema_validation_failures": schema_failures,
        },
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 66)
    print("📊 KPI DEĞERLENDİRME RAPORU  (TEKNOFEST TYDA 3. Senaryo, Md.4)")
    print("=" * 66)
    print(f"Test edilen video : {len(results)} / {len(gt_list)}")
    if failures:
        print(f"Başarısız         : {len(failures)}")
        for f in failures[:5]:
            print(f"   - {f}")
    print("-" * 66)
    print("🎯 ŞARTNAME KPI KALEMLERİ:")
    print(f"  • Olay Tespit Doğruluğu (F1)   : {_mean(m['event_f1']):.3f}")
    print(f"      precision / recall         : {_mean(m['event_precision']):.3f} / {_mean(m['event_recall']):.3f}")
    if m["critical_recall"]:
        print(f"  • Kritik Olay Yakalama Oranı   : {_mean(m['critical_recall']):.3f}  ({len(m['critical_recall'])} videoda)")
    if m["summary"]:
        print(f"  • Özet Kalitesi                : {_mean(m['summary']):.3f}  ({len(m['summary'])} videoda)")
    if m["tool_jaccard"]:
        print(f"  • Aksiyon Önerisi Doğruluğu    : {_mean(m['tool_jaccard']):.3f}")
    if m["risk"]:
        print(f"  • Risk Doğruluğu               : %{_mean(m['risk']) * 100:.1f}")
    print(f"  • GENEL SİSTEM SKORU           : {_mean(m['overall']):.3f}")
    print("-" * 66)
    print("⚡ PERFORMANS VE KAYNAK (Md.4):")
    print(f"  • Ort. İşlem Süresi            : {_mean(m['latency']):.2f} sn")
    if m["rtf"]:
        rt = sum(1 for v in m["rtf"] if v < 1.0)
        print(f"  • Ort. Real Time Factor        : {_mean(m['rtf']):.3f}  ({rt}/{len(m['rtf'])} video gerçek zamandan hızlı)")
    if m["peak_gpu"]:
        print(f"  • Tepe GPU Belleği             : {max(m['peak_gpu']):.2f} GB")
    else:
        print("  • Tepe GPU Belleği             : ölçülemedi (nvidia-smi erişilemiyor)")
    if m["peak_rss"]:
        print(f"  • Tepe Süreç RAM (RSS)         : {max(m['peak_rss']):.2f} GB")
    
    # Özet detaylarını göster
    print("-" * 66)
    print("📝 ÖZET DETAYLARI (ilk 3 video):")
    for i, res in enumerate(results[:3]):
        print(f"\n  Video {i+1}: {res['video_id']}")
        gt_sum = res['summary_details']['ground_truth_summary']
        model_sum = res['summary_details']['model_summary']
        print(f"    • GT Özeti: {gt_sum[:100]}..." if len(gt_sum) > 100 else f"    • GT Özeti: {gt_sum}")
        print(f"    • Model Özeti: {model_sum[:100]}..." if len(model_sum) > 100 else f"    • Model Özeti: {model_sum}")
        print(f"    • Özet Skoru: {res['summary_details']['summary_score']:.3f}")
        
        if res['critical_events']:
            print(f"    • Kritik Olaylar ({len(res['critical_events'])} adet):")
            for crit in res['critical_events'][:2]:
                print(f"      - Zaman: {crit['time']} | {crit['event'][:50]}...")
                print(f"        Eşleşme: {'✅' if crit['is_matched'] else '❌'}")
    
    if schema_failures:
        print("-" * 66)
        print(f"⚠️  ŞEMA DOĞRULAMASI BAŞARISIZ ({len(schema_failures)} video): {', '.join(schema_failures[:5])}")
    if duration_warnings:
        print("-" * 66)
        print("⚠️  SÜRE TUTARSIZLIĞI (ground_truth.json doğrulanmalı):")
        for w in duration_warnings[:5]:
            print(f"   - {w}")
    print("=" * 66)
    print(f"📁 Sonuçlar + metrik tanımları: {output_path}\n")


def print_metric_definitions() -> None:
    print("\n" + "=" * 66)
    print("METRİK TANIMLARI — TEKNOFEST TYDA 3. Senaryo, Madde 4")
    print("=" * 66)
    for name, d in METRIC_DEFINITIONS.items():
        print(f"\n▸ {name}")
        print(f"  Alan      : {d['alan']}")
        print(f"  Tanım     : {d['tanim']}")
        if "esikler" in d:
            print(f"  Eşikler   : {d['esikler']}")
        print(f"  Gerekçe   : {d['gerekce']}")
    print("\n" + "-" * 66)
    print("KATMAN AĞIRLIKLARI:")
    for k, v in LAYER_WEIGHTS.items():
        print(f"  {k:<10} {v:.2f}")
    print(f"  {'TOPLAM':<10} {sum(LAYER_WEIGHTS.values()):.2f}")
    print("=" * 66 + "\n")


# ============================================================================
# 10. SELF-TEST
# ============================================================================

def self_test() -> int:
    failures: List[str] = []

    def check(name: str, cond: bool, extra: str = "") -> None:
        print(f"  {'✅' if cond else '❌'} {name} {extra if not cond else ''}")
        if not cond:
            failures.append(name)

    print("\n[1] Zaman ayrıştırma — sessiz sıfır olmamalı")
    check("'00:12' -> 12", parse_time("00:12") == 12)
    check("'15' -> 15", parse_time("15") == 15)
    check("bozuk -> None", parse_time("abc") is None)
    check("aralık '00:12-00:18'", parse_range("00:12-00:18") == (12, 18))

    print("\n[2] Tolerans — süreye oransal")
    check("7 sn -> 2.0", tolerance_for_duration(7) == 2.0)
    check("140 sn -> 5.0", tolerance_for_duration(140) == 5.0)

    print("\n[3] IoM — uzun GT aralığı cezalandırmamalı")
    long_gt, good_pred = Interval(45, 140), Interval(65, 75)
    check("uzun GT + içeride tahmin -> 1.0", abs(long_gt.iom(good_pred) - 1.0) < 1e-9)

    print("\n[4] FP/FN muhasebesi")
    ev = TemporalEventEvaluator()
    ev.set_tolerance_from_duration(26)
    gt = [
        {"time": "00:04", "start_time": 4, "end_time": 8, "event": "İşçi alana girdi", "is_critical": False},
        {"time": "00:12", "start_time": 12, "end_time": 17, "event": "Kamyon geri manevra yaptı", "is_critical": True},
        {"time": "00:19", "start_time": 18, "end_time": 22, "event": "Kamyon işçiye çarptı", "is_critical": True},
    ]
    perfect = [
        {"time": "00:05", "start_time": "00:05", "end_time": "00:05", "event": "İşçi alana girdi", "phase": "baslangic"},
        {"time": "00:14", "start_time": "00:14", "end_time": "00:14", "event": "Kamyon geri manevra yaptı", "is_critical_moment": True, "phase": "gelisim"},
        {"time": "00:20", "start_time": "00:20", "end_time": "00:20", "event": "Kamyon işçiye çarptı", "is_critical_moment": True, "phase": "sonuc"},
    ]
    r2 = ev.evaluate(perfect, gt, duration_sec=26)
    check(f"mükemmel eşleşme F1={r2['f1_score']}", r2["f1_score"] > 0.95)
    check(f"kritik recall={r2['critical_recall']}", r2["critical_recall"] == 1.0)

    print("\n[5] Hata Toparlanma (Error Recovery) Mantığı")
    bench = EnhancedAgentBenchmarkKPI()
    # 1. Hata yok -> None
    r_no_err = bench.evaluate_recovery({"tool_history": [{"tool": "alert", "result": "ok"}]})
    check("Hata yok -> Recovery None", r_no_err["recovery_score"] is None)

    # 2. Hata var + Sonrasında başarılı çağrı (Retry/Alternatif) -> 1.0
    r_rec_ok = bench.evaluate_recovery({
        "tool_history": [
            {"tool": "alert", "status": "error"},
            {"tool": "alert", "result": {"data": "success"}}
        ]
    })
    check("Hata + Başarılı Retry -> 1.0", r_rec_ok["recovery_score"] == 1.0)

    # 3. Hata var + Sonrasında toparlanamadı / tekrar hata aldı -> 0.0
    r_rec_fail = bench.evaluate_recovery({
        "tool_history": [
            {"tool": "alert", "error": "timeout"},
            {"tool": "alert_alt", "status": "error"}
        ]
    })
    check("Hata + Başarısız Retry -> 0.0", r_rec_fail["recovery_score"] == 0.0)

    # 4. Hata var + Sonrasında hiçbir çağrı yapılmadı -> 0.0
    r_rec_no_retry = bench.evaluate_recovery({
        "tool_history": [
            {"tool": "alert", "error": "failed"}
        ]
    })
    check("Hata + Retry Yok -> 0.0", r_rec_no_retry["recovery_score"] == 0.0)

    print("\n[6] Ağırlıklar toplamı")
    check(f"ağırlık toplamı 1.00 ({sum(LAYER_WEIGHTS.values()):.2f})",
          abs(sum(LAYER_WEIGHTS.values()) - 1.0) < 1e-9)

    print("\n" + "=" * 66)
    if failures:
        print(f"❌ {len(failures)} test başarısız: {', '.join(failures)}")
        return 1
    print("✅ Tüm self-test kontrolleri başarıyla geçti.")
    return 0


# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alfai KPI benchmark (şartname uyumlu)")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--print-metrics", action="store_true",
                        help="Metrik tanımlarını yazdır (Md.4 gereği)")
    parser.add_argument("--ground-truth", default="ground_truth.json")
    parser.add_argument("--videos-dir", default="Videos")
    parser.add_argument("--output", default="benchmark_results_enhanced.json")
    parser.add_argument("--drop-synthetic", action="store_true")
    args = parser.parse_args()

    if args.print_metrics:
        print_metric_definitions()
        sys.exit(0)
    if args.self_test:
        sys.exit(self_test())

    run_enhanced_benchmark(args.ground_truth, args.videos_dir,
                           args.output, args.drop_synthetic)