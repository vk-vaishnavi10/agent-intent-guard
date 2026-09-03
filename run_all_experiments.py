import subprocess, sys, datetime

scripts = ["multi_split_eval.py", "adversarial_test.py", "test_length_evasion.py"]
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outfile = f"final_results_{ts}.txt"

with open(outfile, "w") as f:
    for s in scripts:
        header = f"\n{'='*70}\n=== {s} ===\n{'='*70}\n"
        print(header)
        f.write(header)
        result = subprocess.run([sys.executable, s], capture_output=True, text=True)
        print(result.stdout)
        f.write(result.stdout)
        if result.returncode != 0:
            err = f"\n--- STDERR ({s}) ---\n{result.stderr}\n"
            print(err)
            f.write(err)

print(f"\nSaved combined output to {outfile}")
