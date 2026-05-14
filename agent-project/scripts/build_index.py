import os
import glob

from rag.index import ChromaIndex


def load_corpus():
    files = glob.glob("data/corpus/*.txt") + glob.glob("data/corpus/*.md")
    texts, ids, metas = [], [], []
    for f in files:
        with open(f, "r", encoding="utf-8") as fin:
            txt = fin.read()
        chunks = chunk_text(txt, 200, 40)
        for i, chk in enumerate(chunks):
            texts.append(chk)
            ids.append(f"{os.path.basename(f)}::chunk-{i}")
            metas.append({"source": f, "chunk_id": i})
    return texts, ids, metas


def chunk_text(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


def main():
    idx = ChromaIndex()
    texts, ids, metas = load_corpus()
    idx.add_docs(texts, ids, metas)
    print(f"Added {len(ids)} documents to index")


if __name__ == "__main__":
    main()
