import os
import chromadb
from pathlib import Path

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "200"))
OVERLAP = int(os.environ.get("OVERLAP", "40"))
HEADER = os.environ.get("HEADER", "0") == "1"

def chunk_text(text, size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks

client = chromadb.PersistentClient(path="chroma_db")
try:
    client.delete_collection("aurelia_kb")
except Exception:
    pass
col = client.create_collection("aurelia_kb")

ids, docs, metas = [], [], []
for f in sorted(Path("knowledge_base").glob("*.md")):
    text = f.read_text(encoding="utf-8")
    title = text.splitlines()[0].lstrip("# ").strip()
    for i, ch in enumerate(chunk_text(text, CHUNK_SIZE, OVERLAP)):
        ids.append(f"{f.name}::{i}")
        docs.append(f"{title}\n{ch}" if HEADER else ch)
        metas.append({"source": f.name, "chunk": i})

col.add(ids=ids, documents=docs, metadatas=metas)
print(f"Indexed {len(docs)} chunks from {len(set(m['source'] for m in metas))} files, size={CHUNK_SIZE}, overlap={OVERLAP}, header={HEADER}")
