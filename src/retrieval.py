# retrieval.py
import re
from typing import List, Dict, Any, Tuple
from langchain_community.vectorstores import FAISS

class DocumentRetriever:
    def __init__(self, vectorstore: FAISS):
        self.vectorstore = vectorstore

    def search(self, query: str, k: int = 3, filter: Dict[str, Any] = None) -> List[Tuple[Any, float]]:
        cleaned_query = self._preprocess_query(query)

        try:
            results_with_scores = self.vectorstore.similarity_search_with_score(
                cleaned_query,
                k=k,
                filter=filter
            )
            print(f"Recherche réussie pour: '{cleaned_query}' avec {len(results_with_scores)} résultats")
            return results_with_scores
        except Exception as e:
            print(f"Erreur lors de la recherche: {e}")
            return []

    def _preprocess_query(self, query: str) -> str:
        query = re.sub(r'[^\w\s]', '', query)
        query = re.sub(r'\s+', ' ', query).strip()
        query = query.lower()
        return query

    def format_results(self, results_with_scores: List[Tuple[Any, float]]) -> Dict[str, Any]:
        formatted_results = {
            "documents": [],
            "scores": []
        }

        for doc, score in results_with_scores:
            formatted_results["documents"].append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "snippet": self._generate_snippet(doc.page_content, 150)
            })
            normalized_score = max(0, min(1, 1 - score / 2)) * 100 if score < 2 else 0
            formatted_results["scores"].append(normalized_score)

        return formatted_results  # Comma was added here

    def _generate_snippet(self, text: str, max_length: int = 150) -> str:
        if len(text) <= max_length:
            return text

        snippet = text[:max_length].strip()
        last_space = snippet.rfind(' ')
        if last_space > max_length * 0.8:
            snippet = snippet[:last_space]

        return snippet + "..."