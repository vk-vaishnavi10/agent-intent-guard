import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import recall_score
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer

LEN_CAP = 500

df = pd.read_csv("features.csv")
df["group"] = np.arange(len(df)) // 2
feature_cols = ["memory_len_capped", "injection_keyword_score", "num_foreign_addrs",
                 "has_foreign_addr", "query_memory_similarity", "starts_with_imperative"]

X = df[feature_cols].values
y = df["label"].values
groups = df["group"].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

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

FILLER = "This is additional context provided for reference and does not affect the requested outcome. "

def features_for(query, memory):
    query_addrs = extract_addrs(query)
    mem_addrs = extract_addrs(memory)
    foreign = mem_addrs - query_addrs
    return {
        "memory_len_capped": min(len(memory), LEN_CAP),
        "injection_keyword_score": injection_score(memory),
        "num_foreign_addrs": len(foreign),
        "has_foreign_addr": int(len(foreign) > 0),
        "query_memory_similarity": max_query_similarity(query, memory),
        "starts_with_imperative": starts_with_imperative(memory),
    }

def combined_memory(memory, pad_repeats):
    m = paraphrase(memory)
    return m + ((" " + FILLER * pad_repeats) if pad_repeats else "")

attack_train = train_df[train_df["label"] == 1]
AUG_PAD_LEVEL = 50
aug_rows = []
for _, row in attack_train.iterrows():
    feats = features_for(row["query"], combined_memory(row["memory"], AUG_PAD_LEVEL))
    feats["label"] = 1
    aug_rows.append(feats)
aug_df = pd.DataFrame(aug_rows)

X_train_orig, y_train_orig = train_df[feature_cols].values, train_df["label"].values
X_train_aug = np.vstack([X_train_orig, aug_df[feature_cols].values])
y_train_aug = np.concatenate([y_train_orig, aug_df["label"].values])

xgb_baseline = XGBClassifier(n_estimators=200, max_depth=4, eval_metric="logloss").fit(X_train_orig, y_train_orig)
xgb_advtrain = XGBClassifier(n_estimators=200, max_depth=4, eval_metric="logloss").fit(X_train_aug, y_train_aug)

attack_test = test_df[test_df["label"] == 1].copy()
y_attack_true = np.ones(len(attack_test))

def build_df(rows_source, transform_fn):
    rows = []
    for _, row in rows_source.iterrows():
        rows.append(features_for(row["query"], transform_fn(row["memory"])))
    return pd.DataFrame(rows)

def recall_of(m, d):
    return recall_score(y_attack_true, m.predict(d[feature_cols].values))

print("\n=== Stress test 1: combined attack at low (pre-saturation) padding levels ===")
print(f"{'pad_x':>6s} {'avg_len':>8s} {'baseline_recall':>16s} {'advtrained_recall':>18s}")
for pad in [0, 1, 2, 3, 5]:
    cdf = build_df(attack_test, lambda mem, pad=pad: combined_memory(mem, pad))
    avg_len = cdf["memory_len_capped"].mean()
    print(f"{pad:>6d} {avg_len:>8.0f} {recall_of(xgb_baseline, cdf):>16.3f} {recall_of(xgb_advtrain, cdf):>18.3f}")

benign_test = test_df[test_df["label"] == 0].copy()

print("\n=== Stress test 2: false positives on padded but GENUINELY BENIGN memory ===")
print(f"{'pad_x':>6s} {'avg_len':>8s} {'baseline_FPR':>13s} {'advtrained_FPR':>15s}")
for pad in [0, 3, 5, 25, 50, 100]:
    bdf = build_df(benign_test, lambda mem, pad=pad: (mem + ((" " + FILLER * pad) if pad else "")))
    avg_len = bdf["memory_len_capped"].mean()
    fp_base = xgb_baseline.predict(bdf[feature_cols].values).mean()
    fp_adv = xgb_advtrain.predict(bdf[feature_cols].values).mean()
    print(f"{pad:>6d} {avg_len:>8.0f} {fp_base:>13.4f} {fp_adv:>15.4f}")

print("\n=== Mechanism check: fraction of TRAINING rows sitting exactly at the 500 cap ===")
print(f"Original attack (pre-augmentation): {(train_df[train_df.label==1]['memory_len_capped']==500).mean():.3f}")
print(f"Original benign:                    {(train_df[train_df.label==0]['memory_len_capped']==500).mean():.3f}")
print(f"Augmented (combined-attack) rows:   {(aug_df['memory_len_capped']==500).mean():.3f}")
