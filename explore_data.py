from datasets import load_dataset

ds = load_dataset("SentientAGI/crypto-agent-safe-function-calling", split="train")
print(f"Rows: {len(ds)}")
print(f"Columns: {ds.column_names}")
print()
print("Sample row 0:")
row = ds[0]
for k, v in row.items():
    print(f"--- {k} ---")
    print(str(v)[:300])
    print()
