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

X = df[feature_cols].values
y = df["label"].values
groups = df["group"].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
xgb_full = XGBClassifier(n_estimators=200, max_depth=4, eval_metric="logloss").fit(X_train, y_train)
ablation_cols = [c for c in feature_cols if c != "injection_keyword_score"]
xgb_noKw = XGBClassifier(n_estimators=200, max_depth=4, eval_metric="logloss").fit(
    df.iloc[train_idx][ablation_cols].values, y_train)

REPLACEMENTS = {
    "remember to": "please", "always use": "use", "instead of": "rather than",
    "reroute": "send", "redirect": "direct", "phishing": "special",
    "hacker": "backup", "stealth": "quiet", "override": "update",
    "for safekeeping": "for storage", "secure handling": "careful handling",
    "malicious": "alternate", "ignore previous": "disregard earlier",
}

def paraphrase(text):
    t = text
    for old, new in REPLACEMENTS.items():
        t = re.sub(re.escape(old), new, t, flags=re.IGNORECASE)
    return t

ADDR_RE = re.compile(r"0x[a-fA-F0-9]{40}")
ADDR_RE_BTC = re.compile(r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b")
ADDR_RE_BECH32 = re.compile(r"\bbc1[a-z0-9]{25,90}\b")
ADDR_LABEL_RE = re.compile(r"address[:\s]+['\"]?([A-Za-z0-9]{6,})['\"]?", re.IGNORECASE)
INJECTION_PHRASES = list(REPLACEMENTS.keys())
IMPERATIVE_STARTERS = {
    "use", "always", "remember", "send", "transfer", "route", "reroute",
    "redirect", "override", "avoid", "ensure", "verify", "call", "switch",
    "replace", "prefer", "direct", "forward", "do", "don't", "never",
    "make", "set", "change", "update", "check", "confirm", "add",
    "remove", "delete", "please", "instead",
}

def injection_score(text):
    t = (text or "").lower()
    return sum(1 for p in INJECTION_PHRASES if p in t)

def extract_addrs(text):
    t = text or ""
    addrs = set(ADDR_RE.findall(t))
    addrs |= set(ADDR_RE_BTC.findall(t))
    addrs |= set(ADDR_RE_BECH32.findall(t))
    addrs |= set(ADDR_LABEL_RE.findall(t))
    return addrs

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
print(f"Attack rows in test set: {len(attack_test)}")

para_rows = []
for _, r in attack_test.iterrows():
    new_memory = paraphrase(r["memory"])
    query_addrs = extract_addrs(r["query"])
    mem_addrs = extract_addrs(new_memory)
    foreign = mem_addrs - query_addrs
    sim = max_query_similarity(r["query"], new_memory)
    para_rows.append({
        "memory_len_capped": min(len(new_memory), LEN_CAP),
        "injection_keyword_score": injection_score(new_memory),
        "num_foreign_addrs": len(foreign),
        "has_foreign_addr": int(len(foreign) > 0),
        "query_memory_similarity": sim,
        "starts_with_imperative": starts_with_imperative(new_memory),
    })

para_df = pd.DataFrame(para_rows)
y_attack_true = np.ones(len(para_df))

def recall_report(name, model_obj, cols, X_orig_attack):
    r_orig = recall_score(y_attack_true, model_obj.predict(X_orig_attack))
    r_para = recall_score(y_attack_true, model_obj.predict(para_df[cols].values))
    print(f"{name:35s} recall on original attacks={r_orig:.3f}   recall on PARAPHRASED attacks={r_para:.3f}")

orig_attack_X = attack_test[feature_cols].values
orig_attack_X_noKw = attack_test[ablation_cols].values

recall_report("Logistic regression", lr, feature_cols, orig_attack_X)
recall_report("XGBoost (full features)", xgb_full, feature_cols, orig_attack_X)
recall_report("XGBoost (no keyword feature)", xgb_noKw, ablation_cols, orig_attack_X_noKw)

rule_orig = (attack_test["injection_keyword_score"] > 0).astype(int).values
rule_para = (para_df["injection_keyword_score"] > 0).astype(int).values
print(f"{'Baseline: keyword rule only':35s} recall on original attacks={rule_orig.mean():.3f}   recall on PARAPHRASED attacks={rule_para.mean():.3f}")
