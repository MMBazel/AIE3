from collections import defaultdict
import numpy as np
from typing import List, Tuple, Callable
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio

def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)

def euclidean_distance(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the Euclidean distance between two vectors."""
    # Change 1: Ensure vectors are numpy arrays
    return np.linalg.norm(np.array(vector_a) - np.array(vector_b))

class VectorDatabase:
    def __init__(self, embedding_model=None):
        # Change 2: Update to store metadata
        self.vectors = defaultdict(lambda: {"vector": None, "metadata": {}})
        self.embedding_model = embedding_model or EmbeddingModel()

    def insert(self, key: str, vector: np.array, metadata: dict = None) -> None:
        # Change 3: Ensure vectors are numpy arrays
        self.vectors[key] = {"vector": np.array(vector), "metadata": metadata or {}}

    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Callable = cosine_similarity,
    ) -> List[Tuple[str, float]]:
        # Change 4: Ensure query_vector is a numpy array
        query_vector = np.array(query_vector)
        scores = [
            # Change 5: Ensure vectors in calculations are numpy arrays
            (key, distance_measure(query_vector, np.array(value["vector"])))
            for key, value in self.vectors.items()
        ]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = cosine_similarity,
        return_as_text: bool = False,
    ) -> List[Tuple[str, float]]:
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(query_vector, k, distance_measure)
        return [result[0] for result in results] if return_as_text else results

    def retrieve_from_key(self, key: str) -> dict:
        return self.vectors.get(key, None)

    async def abuild_from_list(self, list_of_text: List[str]) -> "VectorDatabase":
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        for text, embedding in zip(list_of_text, embeddings):
            self.insert(text, np.array(embedding))
        return self

async def main():
    list_of_text = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]

    vector_db = VectorDatabase()
    await vector_db.abuild_from_list(list_of_text)

    # Insert metadata
    for text in list_of_text:
        metadata = {"source": "example_text", "length": len(text)}
        vector_db.insert(text, vector_db.embedding_model.get_embedding(text), metadata)

    k = 2

    # Perform a search and print results
    searched_vector = vector_db.search_by_text("I think fruit is awesome!", k=k, distance_measure=euclidean_distance)
    print(f"Closest {k} vector(s):", searched_vector)

    # Retrieve a specific vector and print it with metadata
    retrieved_vector = vector_db.retrieve_from_key("I like to eat broccoli and bananas.")
    print("Retrieved vector with metadata:", retrieved_vector)

    # Perform another search and print results as text
    relevant_texts = vector_db.search_by_text("I think fruit is awesome!", k=k, distance_measure=euclidean_distance, return_as_text=True)
    print(f"Closest {k} text(s):", relevant_texts)

if __name__ == "__main__":
    asyncio.run(main())
