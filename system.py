from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class RAGSystem:
    def __init__(self, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = 384
        self.index = None
        self.documents = []

    def load_documents(self, documents):
        embeddings = self.embedding_model.encode(documents, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings.astype('float32'))
        self.documents = documents

    def retrieve(self, query, top_k=3):
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        similarities, indices = self.index.search(query_embedding.astype('float32'), top_k)
        results = [(self.documents[idx], float(sim)) for idx, sim in zip(indices[0], similarities[0])]
        return results
