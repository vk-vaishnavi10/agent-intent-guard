import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer

LEN_CAP = 500

df = pd.read_csv("features.csv")
df["group"] = np.arange(len(df)) // 2
feature_cols = ["memory_len_capped", "injection_keyword_score", "num_foreign_addrs",
                 "has_foreign_addr", "query_memory_similarity", "starts_with_imperative"]
ablation_cols = [c for c in feature_cols if c != "injection_keyword_score"]

X = df[feature_cols].values
y = df["label"].values
groups = df["group"].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, y_train = X[train_idx], y[train_idx]

lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
xgb_full = XGBClassifier(n_estimators=200, max_depth=4, eval_metric="logloss").fit(X_train, y_train)
xgb_noKw = XGBClassifier(n_estimators=200, max_depth=4, eval_metric="logloss").fit(
    df.iloc[train_idx][ablation_cols].values, y_train)

REPLACEMENTS = {
    "remember to": "please", "always use": "use", "instead of": "rather than",
    "reroute": "send", "redirect": "direct", "phishing": "special",
    "hacker": "backup", "stealth": "quiet", "override": "update",
    "for safekeeping": "for storage", "secure handling": "careful handling",
    "malicious": "alternate", "ignore previous": "disregard earlier",
}
INJECTION_PHRASES = list(REPLACEMENTS.keys())

def paraphrase(text):
    t = text
    for old, new in REPLACEMENTS.items():
        t = re.sub(re.escape(old), new, t, flags=re.IGNORECASE)
    return t

ADDR_RE = re.compile(r"0x[a-fA-F0-9]{40}")
ADDR_RE_BTC = re.compile(r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b")
ADDR_RE_BECH32 = re.compile(r"\bbc1[a-z0-9]{25,90}\b")
ADDR_LABEL_RE = re.compile(r"address[:\s]+['\"]?([A-Za-z0-9]{6,})['\"]?", re.IGNORECASE)
IMPERATIVE_STARTERS = {
    "use", "always", "remember", "send", "transfer", "route", "reroute",
    "redirect", "override", "avoid", "ensure", "verify", "call", "switch",
    "replace", "prefer", "direct", "forward", "do", "don't", "never",
    "make", "set", "change", "update", "check", "confirm", "add",
    "remove", "delete", "please", "instead",
}

def extract_addrs(text):
    t = text or ""
    addrs = set(ADDR_RE.findall(t))
    addrs |= set(ADDR_RE_BTC.findall(t))
    addrs |= set(ADDR_RE_BECH32.findall(t))
    addrs |= set(ADDR_LABEL_RE.findall(t))
    return addrs

def injection_score(text):
    t = (text or "").lower()
    return sum(1 for p in INJECTION_PHRASES if p in t)

def starts_with_imperative(text):
    words = re.findall(r"[a-zA-Z']+", (text or "").lower())
    return int(bool(words) and words[0] in IMPERATIVE_STARTERS)

def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p for p in parts if p]

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

def max_query_similarity(query, memory):
    if not memory:
        return 0.0
    q_emb = model.encode(query)
    q_norm = np.linalg.norm(q_emb) + 1e-9
    sentences = split_sentences(memory) or [memory]
    m_embs = model.encode(sentences)
    sims = [float(np.dot(q_emb, m) / (q_norm * (np.linalg.norm(m) + 1e-9))) for m in m_embs]
    return max(sims)

test_df = df.iloc[test_idx].copy()
attack_test = test_df[test_df["label"] == 1].copy()
print(f"Attack rows in test set: {len(attack_test)}\n")

FILLER = "This is additional context provided for reference and does not affect the requested outcome. "
PAD_REPEATS = [0, 10, 25, 50, 100]

def build_combined_features(pad_repeats):
    rows = []
    for _, r in attack_test.iterrows():
        combined_memory = paraphrase(r["memory"])
        combined_memory = combined_memory + ((" " + FILLER * pad_repeats) if pad_repeats else "")
        query_addrs = extract_addrs(r["query"])
        mem_addrs = extract_addrs(combined_memory)
        foreign = mem_addrs - query_addrs
        sim = max_query_similarity(r["query"], combined_memory)
        rows.append({
            "memory_len_capped": min(len(combined_memory), LEN_CAP),
            "injection_keyword_score": injection_score(combined_memory),
            "num_foreign_addrs": len(foreign),
            "has_foreign_addr": int(len(foreign) > 0),
            "query_memory_similarity": sim,
            "starts_with_imperative": starts_with_imperative(combined_memory),
        })
    return pd.DataFrame(rows)

y_true = np.ones(len(attack_test))

print("Combined attack: paraphrase trigger phrases + pad with filler")
print(f"{'padding_x':>10s}  {'avg_len':>8s}  {'LR':>6s}  {'XGB_full':>9s}  {'XGB_noKw':>9s}  {'KeywordRule':>12s}")
for pad in PAD_REPEATS:
    cdf = build_combined_features(pad)
    avg_len = cdf["memory_len_capped"].mean()
    r_lr = recall_score(y_true, lr.predict(cdf[feature_cols].values))
    r_full = recall_score(y_true, xgb_full.predict(cdf[feature_cols].values))
    r_noKw = recall_score(y_true, xgb_noKw.predict(cdf[ablation_cols].values))
    r_rule = (cdf["injection_keyword_score"] > 0).astype(int).mean()
    print(f"{pad:>10d}  {avg_len:>8.0f}  {r_lr:>6.3f}  {r_full:>9.3f}  {r_noKw:>9.3f}  {r_rule:>12.3f}")
