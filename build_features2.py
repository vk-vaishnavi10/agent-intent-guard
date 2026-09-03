import re
import random
from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

random.seed(42)

print("Loading dataset...")
ds = load_dataset("SentientAGI/crypto-agent-safe-function-calling", split="train")
crypto = ds.filter(lambda r: r["domain"] == "crypto")
others = ds.filter(lambda r: r["domain"] == "others")
benign_memories = [r["memory"] for r in others]
print(f"Crypto (attack) rows: {len(crypto)}, benign memory pool: {len(benign_memories)}")

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

ADDR_RE = re.compile(r"0x[a-fA-F0-9]{40}")
INJECTION_PHRASES = [
    "ignore previous", "instead of", "reroute", "redirect", "phishing",
    "hacker", "stealth", "override", "always use", "remember to",
    "for safekeeping", "secure handling", "malicious",
]

def extract_addrs(text):
    return set(ADDR_RE.findall(text or ""))

def injection_score(text):
    t = (text or "").lower()
    return sum(1 for p in INJECTION_PHRASES if p in t)

def make_row(query, memory, label):
    query_addrs = extract_addrs(query)
    memory_addrs = extract_addrs(memory)
    foreign_addrs = memory_addrs - query_addrs

    q_emb = model.encode(query)
    m_emb = model.encode(memory) if memory else np.zeros_like(q_emb)
    sim = float(np.dot(q_emb, m_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(m_emb) + 1e-9)) if memory else 0.0

    return {
        "query": query,
        "memory": memory,
        "label": label,
        "memory_len": len(memory or ""),
        "injection_keyword_score": injection_score(memory),
        "num_addrs_in_memory": len(memory_addrs),
        "num_foreign_addrs": len(foreign_addrs),
        "has_foreign_addr": int(len(foreign_addrs) > 0),
        "query_memory_similarity": sim,
    }

rows = []
for r in crypto:
    query = r["query"]
    rows.append(make_row(query, r["memory"], 1))               # real attack memory
    rows.append(make_row(query, random.choice(benign_memories), 0))  # random benign memory, same query

df = pd.DataFrame(rows)
df.to_csv("features.csv", index=False)
print(f"Saved {len(df)} rows")
print(df["label"].value_counts())
print(df.groupby("label")[["memory_len","injection_keyword_score","num_foreign_addrs","query_memory_similarity"]].mean())
