import sys
import chromadb

col = chromadb.PersistentClient(path="chroma_db").get_collection("aurelia_kb")
question = " ".join(sys.argv[1:])
res = col.query(query_texts=[question], n_results=3)

for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
    print(f"[{meta['source']} | chunk {meta['chunk']} | distance {dist:.3f}]")
    print(doc)
    print()
