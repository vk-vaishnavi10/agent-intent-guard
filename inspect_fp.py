import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

feature_cols = ["memory_len", "injection_keyword_score", "num_foreign_addrs",
                 "has_foreign_addr", "query_memory_similarity"]
ablation_cols = [c for c in feature_cols if c != "injection_keyword_score"]

df = pd.read_csv("features.csv")
df["group"] = np.arange(len(df)) // 2
X = df[feature_cols].values
y = df["label"].values
groups = df["group"].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, _ = next(gss.split(X, y, groups))
Xa_train = df.iloc[train_idx][ablation_cols].values
y_train = y[train_idx]

xgb_noKw = XGBClassifier(n_estimators=200, max_depth=4, eval_metric="logloss")
xgb_noKw.fit(Xa_train, y_train)

diag = pd.read_csv("indomain_benign.csv")
Xa_diag = diag[ablation_cols].values
diag["pred"] = xgb_noKw.predict(Xa_diag)

fp = diag[diag["pred"] == 1]
print(f"False positives: {len(fp)} / {len(diag)}\n")
print("Sample false-positive memory texts:")
for m in fp["memory"].head(15):
    print(" -", m)

print()
print('Unique false-positive templates and counts:')
print(fp['memory'].value_counts())
