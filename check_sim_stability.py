import pandas as pd, numpy as np, re
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import GroupShuffleSplit

df = pd.read_csv('features.csv')
df['group'] = np.arange(len(df)) // 2
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
_, test_idx = next(gss.split(df[['memory_len_capped']].values, df['label'].values, df['group'].values))
test_df = df.iloc[test_idx]
attack_test = test_df[test_df['label']==1].copy()

model = SentenceTransformer('all-MiniLM-L6-v2')
FILLER = 'This is additional context provided for reference and does not affect the requested outcome. '

def split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', (text or '').strip())
    return [p for p in parts if p]

def max_sim(query, memory):
    q = model.encode(query); qn = np.linalg.norm(q) + 1e-9
    sents = split_sentences(memory) or [memory]
    ms = model.encode(sents)
    return max(float(np.dot(q,m)/(qn*(np.linalg.norm(m)+1e-9))) for m in ms)

for pad in [0, 10, 25, 50, 100]:
    sims = []
    for _, r in attack_test.head(50).iterrows():
        padded = r['memory'] + ((' ' + FILLER*pad) if pad else '')
        sims.append(max_sim(r['query'], padded))
    print(f'pad={pad:4d}  mean_max_sentence_similarity={np.mean(sims):.4f}')
