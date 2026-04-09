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

    def query(self, query: str, top_k: int = 3):
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
