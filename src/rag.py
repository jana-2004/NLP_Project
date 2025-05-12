"""
Improved RAG (Retrieval-Augmented Generation) module for mental health advice.
"""

from typing import List
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import torch


class AdviceRAG:
    """
    Retrieval-Augmented Generator for mental health advice.
    Uses FAISS for retrieval and TinyLlama (or any text-generation model) for response generation.
    """

    def __init__(self, advice_path: str, model_name: str = "all-mpnet-base-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.advice_path = advice_path

        # Load advice corpus
        self.advices = self._load_advice(advice_path)

        # Build FAISS index
        self.index, self.embeddings = self._build_faiss_index(self.advices)

        # Load lightweight generator model
        self.generator = pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

    def _load_advice(self, filepath: str) -> List[str]:
        """
        Load advice lines from a text file.
        """
        with open(filepath, "r", encoding="utf-8") as file:
            return [line.strip() for line in file.readlines() if line.strip()]

    def _build_faiss_index(self, texts: List[str]):
        """
        Build FAISS index from list of advice texts.
        """
        embeddings = self.embedder.encode(texts, convert_to_tensor=False, show_progress_bar=True)
        dim = len(embeddings[0])
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index, embeddings

    def retrieve(self, query: str, predicted_class: str, top_k: int = 3) -> str:
        """
        Combine the predicted class with the user query, retrieve top-k relevant advice chunks,
        and generate a human-like response.
        """
        # Enrich the query to improve retrieval
        combined_query = f"Concern: {predicted_class}. User says: {query}"
        query_vec = self.embedder.encode([combined_query])
        D, I = self.index.search(query_vec, top_k)

        # Retrieve top-k chunks
        top_chunks = [self.advices[i] for i in I[0]]
        prompt = self._construct_prompt(query, predicted_class, top_chunks)

        # Generate response
        try:
            generated = self.generator(prompt, max_new_tokens=100, num_return_sequences=1)
            output_text = generated[0]['generated_text']
            response = output_text.split("Response:")[-1].strip()

            # Fallback: if model fails, return top advice
            if not response or len(response.split()) < 5:
                return top_chunks[0]

            return response

        except Exception as e:
            print(f"[RAG ERROR]: {e}")
            return top_chunks[0]  # fallback to first retrieved advice

    def _construct_prompt(self, query: str, predicted_class: str, chunks: List[str]) -> str:
        """
        Construct a prompt for generation using the retrieved chunks.
        """
        joined_chunks = "\n".join(f"- {chunk}" for chunk in chunks)
        return (
            f"The user is experiencing a {predicted_class} issue and says: \"{query}\"\n\n"
            f"Use the following expert advice to write a helpful, empathetic response:\n"
            f"{joined_chunks}\n\n"
            f"Response:"
        )
