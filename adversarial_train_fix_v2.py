import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
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

def pad_only(memory, pad_repeats):
    return memory + ((" " + FILLER * pad_repeats) if pad_repeats else "")

def combined_memory(memory, pad_repeats):
    return pad_only(paraphrase(memory), pad_repeats)

attack_train = train_df[train_df["label"] == 1]
benign_train = train_df[train_df["label"] == 0]
AUG_PAD_LEVELS = [3, 50]

print("Building SYMMETRIC augmentation (both classes padded, not just attack)...")
aug_rows = []
for _, row in attack_train.iterrows():
    for pad in AUG_PAD_LEVELS:
        feats = features_for(row["query"], combined_memory(row["memory"], pad))
        feats["label"] = 1
        aug_rows.append(feats)
for _, row in benign_train.iterrows():
    for pad in AUG_PAD_LEVELS:
        feats = features_for(row["query"], pad_only(row["memory"], pad))
        feats["label"] = 0
        aug_rows.append(feats)
aug_df = pd.DataFrame(aug_rows)
print(f"Augmented rows added: {len(aug_df)}  (attack={int((aug_df.label==1).sum())}, benign={int((aug_df.label==0).sum())})")

X_train_orig, y_train_orig = train_df[feature_cols].values, train_df["label"].values
X_train_v2 = np.vstack([X_train_orig, aug_df[feature_cols].values])
y_train_v2 = np.concatenate([y_train_orig, aug_df["label"].values])
print(f"Training set: original={len(X_train_orig)}  symmetric-augmented={len(X_train_v2)}")

xgb_baseline = XGBClassifier(n_estimators=200, max_depth=4, eval_metric="logloss").fit(X_train_orig, y_train_orig)
xgb_v2 = XGBClassifier(n_estimators=200, max_depth=4, eval_metric="logloss").fit(X_train_v2, y_train_v2)

attack_test = test_df[test_df["label"] == 1].copy()
benign_test = test_df[test_df["label"] == 0].copy()
y_attack_true = np.ones(len(attack_test))
X_test, y_test = test_df[feature_cols].values, test_df["label"].values

def build_df(rows_source, transform_fn):
    rows = []
    for _, row in rows_source.iterrows():
        rows.append(features_for(row["query"], transform_fn(row["memory"])))
    return pd.DataFrame(rows)

def recall_of(m, d):
    return recall_score(y_attack_true, m.predict(d[feature_cols].values))

def eval_clean(m, name):
    pred = m.predict(X_test)
    proba = m.predict_proba(X_test)[:, 1]
    print(f"{name:22s} precision={precision_score(y_test,pred):.3f}  recall={recall_score(y_test,pred):.3f}  "
          f"f1={f1_score(y_test,pred):.3f}  auc={roc_auc_score(y_test,proba):.3f}")

print("\n=== Clean performance (held-out test set) ===")
eval_clean(xgb_baseline, "baseline")
eval_clean(xgb_v2, "symmetric-adv-trained")

print("\n=== Combined-attack recall across padding levels (held-out test set) ===")
print(f"{'pad_x':>6s} {'avg_len':>8s} {'baseline':>10s} {'symmetric_v2':>13s}")
for pad in [0, 1, 2, 3, 5, 25, 50, 100]:
    cdf = build_df(attack_test, lambda mem, pad=pad: combined_memory(mem, pad))
    avg_len = cdf["memory_len_capped"].mean()
    print(f"{pad:>6d} {avg_len:>8.0f} {recall_of(xgb_baseline, cdf):>10.3f} {recall_of(xgb_v2, cdf):>13.3f}")

print("\n=== False positives on padded but GENUINELY BENIGN memory (held-out test set) ===")
print(f"{'pad_x':>6s} {'avg_len':>8s} {'baseline_FPR':>13s} {'symmetric_v2_FPR':>17s}")
for pad in [0, 3, 5, 25, 50, 100]:
    bdf = build_df(benign_test, lambda mem, pad=pad: pad_only(mem, pad))
    avg_len = bdf["memory_len_capped"].mean()
    fp_base = xgb_baseline.predict(bdf[feature_cols].values).mean()
    fp_v2 = xgb_v2.predict(bdf[feature_cols].values).mean()
    print(f"{pad:>6d} {avg_len:>8.0f} {fp_base:>13.4f} {fp_v2:>17.4f}")

print("\n=== Mechanism check: fraction of TRAINING rows sitting exactly at the 500 cap, by class ===")
print(f"Augmented attack rows at cap: {(aug_df[aug_df.label==1]['memory_len_capped']==500).mean():.3f}")
print(f"Augmented benign rows at cap: {(aug_df[aug_df.label==0]['memory_len_capped']==500).mean():.3f}")

print("\n=== In-domain false-positive check (indomain_benign.csv, unmodified) ===")
try:
    ind = pd.read_csv("indomain_benign.csv")
    Xind = ind[feature_cols].values
    print(f"FPR baseline={xgb_baseline.predict(Xind).mean():.4f}   FPR symmetric_v2={xgb_v2.predict(Xind).mean():.4f}")
except FileNotFoundError:
    print("indomain_benign.csv not found - skipping")

# ============================================================
# OOD GENERALIZATION TEST — held-out, hand-authored, never trained on
# ============================================================
print("\n" + "="*60)
print("OOD GENERALIZATION TEST (ood_generalization_set.csv)")
print("="*60)

ood_df = pd.read_csv("ood_generalization_set.csv")

def compute_row_features(query, memory):
    addrs = extract_addrs(memory)
    return {
        "memory_len_capped": min(len(memory), 500),
        "injection_keyword_score": injection_score(memory),
        "num_foreign_addrs": len(addrs),
        "has_foreign_addr": 1 if len(addrs) > 0 else 0,
        "query_memory_similarity": max_query_similarity(query, memory),
        "starts_with_imperative": starts_with_imperative(memory),
    }

ood_feats = ood_df.apply(lambda r: compute_row_features(r["query"], r["memory"]), axis=1)
ood_feats = pd.DataFrame(list(ood_feats))
for col in feature_cols:
    ood_df[col] = ood_feats[col]

print("\nSanity check (should be high for address_redirection, low elsewhere):")
print(ood_df.groupby("subtype")[["has_foreign_addr", "injection_keyword_score", "query_memory_similarity"]].mean())

X_ood = ood_df[feature_cols]
y_ood = ood_df["label"]

for name, model in [("baseline", xgb_baseline), ("v2_symmetric_adv", xgb_v2)]:
    preds = model.predict(X_ood)
    ood_df[f"pred_{name}"] = preds

    benign_mask = y_ood == 0
    attack_mask = y_ood == 1
    fpr = (preds[benign_mask] == 1).mean()
    recall = (preds[attack_mask] == 1).mean()

    print(f"\n--- {name} ---")
    print(f"Overall attack recall: {recall:.4f}  ({attack_mask.sum()} attack rows)")
    print(f"Overall benign FPR:    {fpr:.4f}  ({benign_mask.sum()} benign rows)")
    print("Per-subtype recall:")
    for subtype in ["address_redirection", "tool_substitution", "security_framing", "amount_manipulation"]:
        sub_mask = (ood_df["subtype"] == subtype)
        n = sub_mask.sum()
        r = (ood_df.loc[sub_mask, f"pred_{name}"] == 1).mean() if n else float("nan")
        print(f"  {subtype:22s} n={n:2d}  recall={r:.4f}")

ood_df.to_csv("ood_results.csv", index=False)
print("\nSaved per-row predictions to ood_results.csv")
