import csv
import sys
import time
from collections import defaultdict
from rag import retrieve, ask_llm

label = sys.argv[1] if len(sys.argv) > 1 else "baseline"
out_path = f"results_{label}.csv"

def call_with_retry(question, chunks, tries=3):
    for attempt in range(tries):
        try:
            return ask_llm(question, chunks)
        except Exception as e:
            if attempt == tries - 1:
                return f"ERROR: {e}"
            time.sleep(5)

rows = list(csv.DictReader(open("test_set.csv", encoding="utf-8-sig")))
results = []

for r in rows:
    chunks = retrieve(r["question"])
    sources = []
    for c in chunks:
        if c["source"] not in sources:
            sources.append(c["source"])
    expected = [s.strip() for s in r["source_doc"].split("+") if s.strip()]
    hit = "" if not expected else int(all(s in sources for s in expected))
    answer = call_with_retry(r["question"], chunks)
    results.append({
        "id": r["id"],
        "category": r["category"],
        "question": r["question"],
        "expected_answer": r["expected_answer"],
        "expected_source": r["source_doc"],
        "retrieved_sources": " | ".join(sources),
        "retrieval_hit": hit,
        "best_distance": round(chunks[0]["distance"], 3),
        "bot_answer": answer,
        "correct": "",
        "failure_type": "",
    })
    print(f'{r["id"]:>2} {r["category"]:<12} hit={hit} dist={chunks[0]["distance"]:.2f}')
    time.sleep(1)

with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)

stat = defaultdict(lambda: [0, 0])
dist = defaultdict(list)
for x in results:
    dist[x["category"]].append(x["best_distance"])
    if x["retrieval_hit"] != "":
        stat[x["category"]][0] += x["retrieval_hit"]
        stat[x["category"]][1] += 1

print()
for cat, (h, n) in stat.items():
    print(f"{cat}: retrieval hit {h}/{n}")
for cat, d in dist.items():
    print(f"{cat}: best distance min {min(d):.2f} / avg {sum(d)/len(d):.2f} / max {max(d):.2f}")
print(f"Saved to {out_path}")
