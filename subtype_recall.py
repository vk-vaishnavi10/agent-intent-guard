import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from xgboost import XGBClassifier

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

test_df = df.iloc[test_idx].copy()
attack_test = test_df[test_df["label"] == 1].copy()
print(f"Attack rows in test set: {len(attack_test)}")

TOOL_PATTERNS = [
    "only call", "call the", "function only", "other functions", "other tools",
    "do not use", "don't use", "avoid using", "exclusively use", "must use only",
    "compromised", "no longer available", "instead use", "switch to",
]
SECURITY_PATTERNS = [
    "verify", "security", "secure", "trust", "authenticate",
    "confirm", "validate", "safety", "before fetching", "before proceeding", "check",
]

def classify_subtype(row):
    if row["has_foreign_addr"] == 1:
        return "address_redirection"
    t = (row["memory"] or "").lower()
    tool_hit = any(p in t for p in TOOL_PATTERNS) and ("function" in t or "tool" in t or "call" in t)
    if tool_hit:
        return "tool_substitution"
    if any(p in t for p in SECURITY_PATTERNS):
        return "security_framing"
    return "other_unclassified"

attack_test["subtype"] = attack_test.apply(classify_subtype, axis=1)
print("\nSubtype distribution (heuristic classification, NOT ground truth):")
print(attack_test["subtype"].value_counts())

print("\n=== Recall per attack subtype (held-out test set) ===")
print(f"{'Subtype':22s} {'n':>5s} {'LR':>7s} {'XGB_full':>9s} {'XGB_noKw':>9s} {'KeywordRule':>12s}")
for subtype, sub_df in attack_test.groupby("subtype"):
    n = len(sub_df)
    y_true = np.ones(n)
    r_lr = recall_score(y_true, lr.predict(sub_df[feature_cols].values))
    r_full = recall_score(y_true, xgb_full.predict(sub_df[feature_cols].values))
    r_noKw = recall_score(y_true, xgb_noKw.predict(sub_df[ablation_cols].values))
    r_rule = (sub_df["injection_keyword_score"] > 0).astype(int).mean()
    print(f"{subtype:22s} {n:>5d} {r_lr:>7.3f} {r_full:>9.3f} {r_noKw:>9.3f} {r_rule:>12.3f}")
