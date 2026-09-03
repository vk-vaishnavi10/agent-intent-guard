import pandas as pd, numpy as np, shap, xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

df = pd.read_csv('features.csv')
df['group'] = np.arange(len(df)) // 2

feature_cols = [
    "memory_len_capped", "injection_keyword_score", "num_foreign_addrs",
    "has_foreign_addr", "query_memory_similarity", "starts_with_imperative",
]

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df[feature_cols].values, df['label'].values, df['group'].values))
train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

X_train, y_train = train_df[feature_cols], train_df['label']
X_test, y_test = test_df[feature_cols], test_df['label']

model = xgb.XGBClassifier(n_estimators=200, max_depth=4, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

print("=== Global feature importance (mean |SHAP value|) ===")
mean_abs = np.abs(shap_values.values).mean(axis=0)
order = np.argsort(mean_abs)[::-1]
for i in order:
    print(f"{feature_cols[i]:28s} {mean_abs[i]:.4f}")

print("\n=== Worked example: a correctly-flagged attack instance ===")
preds = model.predict(X_test)
tp_mask = (y_test.values == 1) & (preds == 1)
tp_idx = np.where(tp_mask)[0]
if len(tp_idx) > 0:
    i = tp_idx[0]
    row = X_test.iloc[i]
    sv = shap_values.values[i]
    print("Feature values:")
    for f, v in zip(feature_cols, row):
        print(f"  {f:28s} {v}")
    print("SHAP contributions (sorted by |impact|):")
    for j in np.argsort(np.abs(sv))[::-1]:
        print(f"  {feature_cols[j]:28s} {sv[j]:+.4f}")
    print(f"base_value={explainer.expected_value:.4f}  final_score={explainer.expected_value + sv.sum():.4f}")
else:
    print("No true-positive attack instance found in this split.")
