from typing import List

class DocumentRetriever:
    def __init__(self, faiss_index):
        self.faiss_index = faiss_index

    def retrieve_documents(self, query: str, top_k: int = 3) -> List[str]:
        docs_with_scores = self.faiss_index.similarity_search_with_score(query, k=top_k)
        relevant_docs = [doc.page_content for doc, score in docs_with_scores]
        return relevant_docs

