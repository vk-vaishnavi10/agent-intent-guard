from datasets import load_dataset

ds = load_dataset("SentientAGI/crypto-agent-safe-function-calling", split="train")
others = ds.filter(lambda r: r["domain"] == "others")

for i in range(3):
    print("QUERY:", others[i]["query"][:200])
    print("MEMORY:", others[i]["memory"][:200])
    print("ATTACK LABEL:", others[i]["attack"])
    print("---")
