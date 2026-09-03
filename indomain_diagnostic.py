import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

feature_cols = ["memory_len_capped", "injection_keyword_score", "num_foreign_addrs",
                 "has_foreign_addr", "query_memory_similarity", "starts_with_imperative"]
ablation_cols = [c for c in feature_cols if c != "injection_keyword_score"]

df = pd.read_csv("features.csv")
df["group"] = np.arange(len(df)) // 2
X = df[feature_cols].values
y = df["label"].values
groups = df["group"].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, y_train = X[train_idx], y[train_idx]
Xa_train = df.iloc[train_idx][ablation_cols].values

lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)

xgb_full = XGBClassifier(n_estimators=200, max_depth=4, eval_metric="logloss")
xgb_full.fit(X_train, y_train)

xgb_noKw = XGBClassifier(n_estimators=200, max_depth=4, eval_metric="logloss")
xgb_noKw.fit(Xa_train, y_train)

diag = pd.read_csv("indomain_benign.csv")
assert (diag["label"] == 0).all()

X_diag = diag[feature_cols].values
Xa_diag = diag[ablation_cols].values
rule_pred = (diag["injection_keyword_score"] > 0).astype(int).values

def fpr(name, y_pred):
    fp = int((y_pred == 1).sum())
    n = len(y_pred)
    print(f"{name:35s} false-positive rate on genuine in-domain benign = {fp}/{n} = {fp/n:.3f}")

print(f"In-domain benign diagnostic rows: {len(diag)}\n")
fpr("Baseline: keyword rule only", rule_pred)
fpr("Logistic regression", lr.predict(X_diag))
fpr("XGBoost (full features)", xgb_full.predict(X_diag))
fpr("XGBoost (no keyword feature)", xgb_noKw.predict(Xa_diag))
