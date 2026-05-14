import chromadb

from chromadb.utils import embedding_functions


class ChromaIndex:
    def __init__(
        self, persist_directory: str = "data/index", model_name: str = "BAAI/bge-m3"
    ):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )
        )
        self.collection = self.client.get_or_create_collection(
            name="documents", embedding_function=self.embedding_function
        )

    def add_docs(self, documents: list[str], ids: list[str], metadatas=None):
        self.collection.add(
            ids=ids, documents=documents, metadatas=metadatas or [{}] * len(documents)
        )

    def get_by_ids(self, ids: list[str]):
        """
        Fetch documents/metadatas by ids.
        Returns list of {"id","text","metadata"} for found ids only, or {"error": "..."}.
        """
        try:
            res = self.collection.get(ids=ids, include=["documents", "metadatas"])
            out = []
            got_ids = res.get("ids") or []
            docs = res.get("documents") or []
            metas = res.get("metadatas") or []
            for i, doc_id in enumerate(got_ids):
                doc = docs[i] if i < len(docs) else ""
                meta = metas[i] if i < len(metas) else {}
                out.append({"id": doc_id, "text": doc, "metadata": meta})
            return out
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def query(self, query: str, top_k: int = 10):
        res = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        hits = [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(docs, metas, dists)
        ]
        return hits
