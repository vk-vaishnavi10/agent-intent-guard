import re
import random
from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

random.seed(42)
LEN_CAP = 500

print("Loading dataset...")
ds = load_dataset("SentientAGI/crypto-agent-safe-function-calling", split="train")
crypto = ds.filter(lambda r: r["domain"] == "crypto")
others = ds.filter(lambda r: r["domain"] == "others")
benign_memories = [r["memory"] for r in others]

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

ADDR_RE = re.compile(r"0x[a-fA-F0-9]{40}")
ADDR_RE_BTC = re.compile(r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b")
ADDR_RE_BECH32 = re.compile(r"\bbc1[a-z0-9]{25,90}\b")
ADDR_LABEL_RE = re.compile(r"address[:\s]+['\"]?([A-Za-z0-9]{6,})['\"]?", re.IGNORECASE)
INJECTION_PHRASES = [
    "ignore previous", "instead of", "reroute", "redirect", "phishing",
    "hacker", "stealth", "override", "always use", "remember to",
    "for safekeeping", "secure handling", "malicious",
]
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

def max_query_similarity(query, memory):
    if not memory:
        return 0.0
    q_emb = model.encode(query)
    q_norm = np.linalg.norm(q_emb) + 1e-9
    sentences = split_sentences(memory) or [memory]
    m_embs = model.encode(sentences)
    sims = [float(np.dot(q_emb, m) / (q_norm * (np.linalg.norm(m) + 1e-9))) for m in m_embs]
    return max(sims)

def make_row(query, memory, label):
    query_addrs = extract_addrs(query)
    memory_addrs = extract_addrs(memory)
    foreign_addrs = memory_addrs - query_addrs
    sim = max_query_similarity(query, memory)
    raw_len = len(memory or "")
    return {
        "query": query,
        "memory": memory,
        "label": label,
        "memory_len": raw_len,
        "memory_len_capped": min(raw_len, LEN_CAP),
        "injection_keyword_score": injection_score(memory),
        "num_addrs_in_memory": len(memory_addrs),
        "num_foreign_addrs": len(foreign_addrs),
        "has_foreign_addr": int(len(foreign_addrs) > 0),
        "query_memory_similarity": sim,
        "starts_with_imperative": starts_with_imperative(memory),
    }

NEUTRAL_TEMPLATES = [
    "No special routing instructions on file for this wallet.",
    "Standard transaction memory: proceed exactly as the user requested.",
    "Wallet notes: this is the user's primary account, nothing else specified.",
    "Previous session note: user prefers default network fees, no other preferences saved.",
    "No third-party or alternate addresses associated with this session.",
    "Memory: this request matches the user's typical transaction pattern.",
    "Session memory is empty; treat this as a first-time, standard request.",
    "No unusual activity noted for this account in past sessions.",
    "User's transaction history shows consistent, verified behavior.",
    "This account has no flags or special handling notes.",
    "Default settings apply; no custom routing configured.",
    "No pending changes to wallet preferences.",
    "Account memory: last transaction to this address completed successfully, no issues reported.",
]

TOOL_TRUST_TEMPLATES = [
    "Use the verified data provider for this request; results have matched official sources in the past.",
    "This tool has passed integrity checks in prior sessions.",
    "Data source for this query has a consistent track record of accuracy.",
    "No unauthorized tool substitutions needed for this type of request.",
    "Standard verification steps apply; no shortcuts required here.",
    "Query can be answered with the default toolset, no special routing needed.",
]

SELF_CONSISTENT_TEMPLATES = [
    "Confirmed recipient for this session: {addr}. This matches the user's own request.",
    "Saved address on file: {addr}, previously verified by the user.",
    "Wallet note: {addr} is the user's own recorded address for this type of transaction.",
    "Transaction history confirms {addr} as a legitimate, previously used recipient.",
    "No discrepancy found between the requested address {addr} and records on file.",
]

def genuine_benign_memory(query):
    addrs = extract_addrs(query)
    if addrs and random.random() < 0.4:
        addr = random.choice(list(addrs))
        return random.choice(SELF_CONSISTENT_TEMPLATES).format(addr=addr)
    pool = NEUTRAL_TEMPLATES + TOOL_TRUST_TEMPLATES
    return random.choice(pool)

rows = []
for r in crypto:
    query = r["query"]
    rows.append(make_row(query, r["memory"], 1))
    if random.random() < 0.5:
        benign_memory = random.choice(benign_memories)
    else:
        benign_memory = genuine_benign_memory(query)
    rows.append(make_row(query, benign_memory, 0))

df = pd.DataFrame(rows)
df.to_csv("features.csv", index=False)
print(f"Saved {len(df)} rows")
print(df.groupby("label")[["memory_len_capped","injection_keyword_score","query_memory_similarity","starts_with_imperative"]].mean())
