import re
from collections import Counter

STEM = 5
M = {"ı": "i", "ğ": "g", "ü": "u", "ş": "s", "ö": "o", "ç": "c", "Ğ": "g", "Ü": "u", "Ş": "s", "Ö": "o", "Ç": "c"}
SW = {
    "ve", "ile", "icin", "bir", "bu", "su", "da", "de", "ki", "mi", "mu", "ne", "nasil", "neden", "niye", "cunku",
    "ama", "fakat", "ancak", "lakin", "yani", "demek", "soyle", "boyle", "oyle", "simdi", "sonra", "once",
    "tarafindan", "olarak", "girdi", "cikti", "iken", "gibi", "kadar", "daha", "cok", "her", "olan", "olup",
    "ise", "ayrica", "yine", "hem", "description", "type", "event", "time", "zone", "risk", "confidence"
}

def norm(t):
    t = re.sub(r"[İI]", "i", str(t or ""))
    for a, b in M.items():
        t = t.replace(a, b)
    return t.lower().strip()

def words(t):
    return re.findall(r"[a-z]{2,}", norm(t))

def stem_cur(t):   # MEVCUT: stopword at, sonra len>=5 olmayani at
    return [w[:STEM] for w in words(t) if w not in SW and len(w) >= 3 and len(w) >= STEM]

def stem_content(t):  # V1: len filtresi yok
    return [w[:STEM] for w in words(t) if w not in SW and len(w) >= 3]

def stem_full(t):     # V2: stopword de kalsin -> BITISIKLIK KORUNUR
    return [w[:STEM] for w in words(t)]

def ng(t, n):
    if len(t) < n:
        return Counter()
    return Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))

def rn(p, g, n):
    P, G = ng(p, n), ng(g, n)
    pt, gt = sum(P.values()), sum(G.values())
    if not pt or not gt:
        return 0.0
    o = sum((P & G).values())
    pr, rc = o / pt, o / gt
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0

def lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]
