import csv
import sys
import time
from collections import defaultdict
from rag import retrieve, ask_llm, llm, MODEL, TOP_K

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = {a for a in sys.argv[1:] if a.startswith("--")}
USE_LLM = "--no-llm" not in flags
USE_JUDGE = ("--judge" in flags) and USE_LLM
label = args[0] if args else "baseline_v2"
out_path = f"results_{label}.csv"

def call_with_retry(fn, tries=3):
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            if attempt == tries - 1:
                return f"ERROR: {e}"
            time.sleep(5)

def judge(question, expected, bot_answer):
    prompt = (
        "You grade a customer-support bot.\n"
        f"Question: {question}\n"
        f"Expected answer (ground truth): {expected}\n"
        f"Bot answer: {bot_answer}\n\n"
        "Is the bot answer consistent with the expected answer and free of invented facts? "
        "If the expected answer says the information is not covered, the bot must say it does not have that information "
        "and must not invent it. Reply with exactly one character: 1 if correct, 0 if incorrect."
    )
    def run():
        r = llm.chat.completions.create(
            model=MODEL, temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content or ""
    text = call_with_retry(run).strip()
    if text and text[0] in "01":
        return text[0]
    return "?"

rows = list(csv.DictReader(open("test_set.csv", encoding="utf-8-sig")))
results = []

for r in rows:
    chunks = retrieve(r["question"])
    context_text = "\n".join(c["text"] for c in chunks).lower()
    parts = [p for p in r["key_fact"].split("|") if p]
    has_fact = "" if not parts else int(all(p.lower() in context_text for p in parts))
    answer = call_with_retry(lambda: ask_llm(r["question"], chunks)) if USE_LLM else ""
    if USE_JUDGE and not answer.startswith("ERROR"):
        verdict = judge(r["question"], r["expected_answer"], answer)
    else:
        verdict = ""
    results.append({
        "id": r["id"],
        "category": r["category"],
        "question": r["question"],
        "expected_answer": r["expected_answer"],
        "key_fact": r["key_fact"],
        "retrieved_chunks": " | ".join(c["id"] for c in chunks),
        "context_has_fact": has_fact,
        "best_distance": round(chunks[0]["distance"], 3),
        "bot_answer": answer,
        "judge_auto": verdict,
        "correct": "",
        "failure_type": "",
    })
    print(f'{r["id"]:>2} {r["category"]:<12} fact={has_fact} judge={verdict} dist={chunks[0]["distance"]:.2f}')
    time.sleep(1)

with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)

fact = defaultdict(lambda: [0, 0])
jud = defaultdict(lambda: [0, 0])
dist = defaultdict(list)
for x in results:
    dist[x["category"]].append(x["best_distance"])
    if x["context_has_fact"] != "":
        fact[x["category"]][0] += x["context_has_fact"]
        fact[x["category"]][1] += 1
    if x["judge_auto"] in ("0", "1"):
        jud[x["category"]][0] += int(x["judge_auto"])
        jud[x["category"]][1] += 1

errors = sum(1 for x in results if x["bot_answer"].startswith("ERROR"))
print(f"\nSettings: TOP_K={TOP_K}, model={MODEL}, llm={USE_LLM}, judge={USE_JUDGE}")
if errors:
    print(f"WARNING: {errors} answers failed with ERROR (see bot_answer column); generation results are incomplete")
for cat, (h, n) in fact.items():
    print(f"{cat}: context has key fact {h}/{n}")
for cat, (h, n) in jud.items():
    print(f"{cat}: judge says correct {h}/{n}")
for cat, d in dist.items():
    print(f"{cat}: best distance min {min(d):.2f} / avg {sum(d)/len(d):.2f} / max {max(d):.2f}")
print(f"Saved to {out_path}")
