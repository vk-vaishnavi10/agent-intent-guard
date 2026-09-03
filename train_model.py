import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, classification_report
from xgboost import XGBClassifier

df = pd.read_csv("features.csv")

# each pair of rows (attack, benign) came from the same crypto query, in order —
# group by pair index so a query's two versions never split across train/test
df["group"] = np.arange(len(df)) // 2

feature_cols = ["memory_len_capped", "injection_keyword_score", "num_foreign_addrs",
                 "has_foreign_addr", "query_memory_similarity", "starts_with_imperative"]

X = df[feature_cols].values
y = df["label"].values
groups = df["group"].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

def report(name, y_true, y_pred, y_prob=None):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    line = f"{name:35s} precision={p:.3f}  recall={r:.3f}  f1={f1:.3f}"
    if y_prob is not None:
        line += f"  auc={roc_auc_score(y_true, y_prob):.3f}"
    print(line)

print(f"Train: {len(X_train)}  Test: {len(X_test)}\n")

# Baseline 1: dumb rule — flag if any injection keyword present
rule_pred = (df.iloc[test_idx]["injection_keyword_score"] > 0).astype(int).values
report("Baseline: keyword rule only", y_test, rule_pred)

# Baseline 2: logistic regression, full features
lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
report("Baseline: logistic regression", y_test, lr.predict(X_test), lr.predict_proba(X_test)[:,1])

# Model A: XGBoost, full features
xgb_full = XGBClassifier(n_estimators=200, max_depth=4, eval_metric="logloss")
xgb_full.fit(X_train, y_train)
report("XGBoost (full features)", y_test, xgb_full.predict(X_test), xgb_full.predict_proba(X_test)[:,1])

# Model B: XGBoost WITHOUT injection_keyword_score (ablation — proves it's not just keyword-spotting)
ablation_cols = [c for c in feature_cols if c != "injection_keyword_score"]
Xa_train = df.iloc[train_idx][ablation_cols].values
Xa_test = df.iloc[test_idx][ablation_cols].values
xgb_noKw = XGBClassifier(n_estimators=200, max_depth=4, eval_metric="logloss")
xgb_noKw.fit(Xa_train, y_train)
report("XGBoost (no keyword feature)", y_test, xgb_noKw.predict(Xa_test), xgb_noKw.predict_proba(Xa_test)[:,1])

print("\nFeature importance (full model):")
for name, imp in sorted(zip(feature_cols, xgb_full.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:30s} {imp:.3f}")
