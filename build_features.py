import re
import json
from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

print("Loading dataset...")
ds = load_dataset("SentientAGI/crypto-agent-safe-function-calling", split="train")
crypto = ds.filter(lambda r: r["domain"] == "crypto")
print(f"Crypto attack rows: {len(crypto)}")

print("Loading embedding model (first run downloads ~80MB)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

ADDR_RE = re.compile(r"0x[a-fA-F0-9]{40}")
INJECTION_PHRASES = [
    "ignore previous", "instead of", "reroute", "redirect", "phishing",
    "hacker", "stealth", "override", "always use", "remember to",
    "for safekeeping", "secure handling", "malicious",
]

def extract_addrs(text):
    return set(ADDR_RE.findall(text or ""))

def injection_score(memory_text):
    t = (memory_text or "").lower()
    return sum(1 for p in INJECTION_PHRASES if p in t)

rows = []

for r in crypto:
    query = r["query"]
    memory = r["memory"]
    answers = r["answers"]
    tools = r["tools"]

    query_addrs = extract_addrs(query)
    memory_addrs = extract_addrs(memory)
    answer_addrs = extract_addrs(answers)

    # semantic similarity between stated request and the action taken
    q_emb = model.encode(query)
    a_emb = model.encode(answers)
    sim = float(np.dot(q_emb, a_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(a_emb) + 1e-9))

    def make_row(label, mem_text, mem_addrs):
        # core signal: does an address used in the final action appear ONLY in memory,
        # never in the user's original query? that's an injected recipient.
        injected_addr_used = bool(answer_addrs & mem_addrs - query_addrs) if mem_addrs else False
        return {
            "query": query,
            "memory": mem_text,
            "answers": answers,
            "label": label,  # 1 = attack, 0 = benign
            "query_len": len(query),
            "memory_len": len(mem_text or ""),
            "injection_keyword_score": injection_score(mem_text),
            "num_addrs_in_query": len(query_addrs),
            "num_addrs_in_memory": len(mem_addrs),
            "num_addrs_in_answers": len(answer_addrs),
            "injected_addr_used_in_action": int(injected_addr_used),
            "query_action_similarity": sim,
        }

    rows.append(make_row(1, memory, memory_addrs))       # attack version
    rows.append(make_row(0, "", set()))                   # benign counterfactual, no memory

df = pd.DataFrame(rows)
df.to_csv("features.csv", index=False)
print(f"Saved {len(df)} rows to features.csv")
print(df["label"].value_counts())
print(df.groupby("label")["injected_addr_used_in_action"].mean())
