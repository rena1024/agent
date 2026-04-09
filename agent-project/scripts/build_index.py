import os, glob

from rag.index import ChromaIndex


def load_corpus():
    files = glob.glob("data/corpus/*.txt") + glob.glob("data/corpus/*.md")
    texts, ids, metas = [], [], []
    for f in files:
        with open(f, "r", encoding="utf-8") as fin:
            txt = fin.read()
        ids.append(os.path.basename(f))
        texts.append(txt)
        metas.append({"source": f})
    return texts, ids, metas


def main():
    idx = ChromaIndex()
    texts, ids, metas = load_corpus()
    idx.add_docs(texts, ids, metas)
    print(f"Added {len(ids)} documents to index")


if __name__ == "__main__":
    main()
