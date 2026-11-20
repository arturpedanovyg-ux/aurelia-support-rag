import chromadb

col = chromadb.PersistentClient(path="chroma_db").get_collection("aurelia_kb")

tests = [
    (5, "До скольки нужно заказать, чтобы отправили в тот же день?",
        "What is the cutoff time to order for same-day shipping?", "before 14:00 cet"),
    (12, "Скільки коштує гравіювання?",
         "How much does engraving cost?", "engraving costs eur 15"),
]

for qid, original, english, fact in tests:
    for label, q in (("original", original), ("english ", english)):
        res = col.query(query_texts=[q], n_results=10)
        docs = [d.lower() for d in res["documents"][0]]
        rank = next((i + 1 for i, d in enumerate(docs) if fact in d), None)
        print(f"#{qid} {label}: fact at rank {rank if rank else 'NOT FOUND'} | best distance {res['distances'][0][0]:.2f}")
