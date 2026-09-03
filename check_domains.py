from datasets import load_dataset
import json

ds = load_dataset("SentientAGI/crypto-agent-safe-function-calling", split="train")

print("Domain counts:")
from collections import Counter
print(Counter(ds["domain"]))
print()
print("Attack type counts:")
print(Counter(ds["attack"]))
print()

# look for wallet/transfer/send-style function names in answers or tools
transfer_keywords = ["send", "transfer", "withdraw", "swap", "wallet", "recipient", "address"]
count = 0
examples = []
for row in ds:
    text = (row["answers"] + row["tools"]).lower()
    if any(k in text for k in transfer_keywords):
        count += 1
        if len(examples) < 3:
            examples.append(row)

print(f"Rows mentioning transfer-like keywords: {count} / {len(ds)}")
print()
for ex in examples:
    print("QUERY:", ex["query"][:200])
    print("MEMORY:", ex["memory"][:200])
    print("ANSWERS:", ex["answers"][:300])
    print("---")
