import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from xgboost import XGBClassifier

df = pd.read_csv("features.csv")
df["group"] = np.arange(len(df)) // 2
diag = pd.read_csv("indomain_benign.csv")

feature_cols = ["memory_len_capped", "injection_keyword_score", "num_foreign_addrs",
                 "has_foreign_addr", "query_memory_similarity", "starts_with_imperative"]
ablation_cols = [c for c in feature_cols if c != "injection_keyword_score"]

X = df[feature_cols].values
y = df["label"].values
groups = df["group"].values
X_diag = diag[feature_cols].values
Xa_diag = diag[ablation_cols].values

SEEDS = [0, 1, 2, 3, 4, 42, 123, 2024, 7, 99]

results = {name: {"precision": [], "recall": [], "f1": [], "auc": [], "fpr_indomain": []}
           for name in ["keyword_rule", "logreg", "xgb_full", "xgb_noKw"]}

for seed in SEEDS:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    rule_pred = (df.iloc[test_idx]["injection_keyword_score"] > 0).astype(int).values
    p, r, f1, _ = precision_recall_fscore_support(y_test, rule_pred, average="binary")
    rule_fpr = ((diag["injection_keyword_score"] > 0).astype(int).values == 1).mean()
    for k, v in zip(["precision","recall","f1","auc","fpr_indomain"], [p, r, f1, np.nan, rule_fpr]):
        results["keyword_rule"][k].append(v)

    lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    pred, prob = lr.predict(X_test), lr.predict_proba(X_test)[:, 1]
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary")
    fpr = (lr.predict(X_diag) == 1).mean()
    for k, v in zip(["precision","recall","f1","auc","fpr_indomain"], [p, r, f1, roc_auc_score(y_test, prob), fpr]):
        results["logreg"][k].append(v)

    xgb_full = XGBClassifier(n_estimators=200, max_depth=4, eval_metric="logloss").fit(X_train, y_train)
    pred, prob = xgb_full.predict(X_test), xgb_full.predict_proba(X_test)[:, 1]
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary")
    fpr = (xgb_full.predict(X_diag) == 1).mean()
    for k, v in zip(["precision","recall","f1","auc","fpr_indomain"], [p, r, f1, roc_auc_score(y_test, prob), fpr]):
        results["xgb_full"][k].append(v)

    Xa_train, Xa_test = df.iloc[train_idx][ablation_cols].values, df.iloc[test_idx][ablation_cols].values
    xgb_noKw = XGBClassifier(n_estimators=200, max_depth=4, eval_metric="logloss").fit(Xa_train, y_train)
    pred, prob = xgb_noKw.predict(Xa_test), xgb_noKw.predict_proba(Xa_test)[:, 1]
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary")
    fpr = (xgb_noKw.predict(Xa_diag) == 1).mean()
    for k, v in zip(["precision","recall","f1","auc","fpr_indomain"], [p, r, f1, roc_auc_score(y_test, prob), fpr]):
        results["xgb_noKw"][k].append(v)

print(f"Evaluated across {len(SEEDS)} random splits (mean ± std)\n")
for name, metrics in results.items():
    line = f"{name:15s} "
    for metric_name, vals in metrics.items():
        arr = np.array(vals, dtype=float)
        if np.isnan(arr).all():
            continue
        line += f"{metric_name}={arr.mean():.3f}±{arr.std():.3f}  "
    print(line)
