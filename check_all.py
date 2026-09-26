"""Run every file's self-check, print one table, exit nonzero on failure.

CI on github runners is compile-only (no GPU), so this is the real gate —
run it before pushing. Uses each file's exit code rather than scraping its
output; the ✅ lines are for humans, the return code is for scripts.

    python check_all.py            # everything, ~4-5 min (first cuda build longer)
    python check_all.py 08 21 26   # just a few files, any prefix match

benchmark.py is deliberately not in the list (it regenerates the charts);
target.py is a 3-line device probe, not a test.
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = sorted(
    f for f in os.listdir(HERE)
    if f.endswith(".py") and f[0].isdigit()
) + ["vector_add.py", "fused_softmax.py"]


def main():
    wanted = sys.argv[1:]
    files = [f for f in FILES if not wanted or any(f.startswith(w) for w in wanted)]

    results = []
    for f in files:
        t0 = time.time()
        # file 29's first-ever run compiles the cuda extension (~2 min)
        p = subprocess.run([sys.executable, f], capture_output=True, text=True, timeout=600)
        ok = p.returncode == 0
        last = (p.stdout.strip().splitlines() or ["(no output)"])[-1]
        results.append((f, ok, time.time() - t0))
        print(f"{'✅' if ok else '❌'} {f:<32} {time.time() - t0:5.1f}s   {last[:70]}")

    failed = [f for f, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed"
          + (f", failed: {', '.join(failed)}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
