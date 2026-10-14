
from pathlib import Path

from backend.rag.ingest import (
    DEFAULT_DATABASE_DIR,
    create_collection,
)


def search_guidance(
    query: str,
    n_results: int = 4,
    database_dir: Path = DEFAULT_DATABASE_DIR,
) -> list[dict]:
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty")

    if n_results <= 0:
        raise ValueError("n_results must be greater than zero")

    collection = create_collection(Path(database_dir))

    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[query.strip()],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    matches = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for document, metadata, distance in zip(
        documents, metadatas, distances
    ):
        matches.append(
            {
                "text": document,
                "source_filename": metadata["source_filename"],
                "page_number": metadata["page_number"],
                "chunk_index": metadata["chunk_index"],
                "distance": distance,
            }
        )

    return matches


if __name__ == "__main__":
    question = input("What emissions guidance do you want to find? ")

    for match in search_guidance(question):
        print(f"\nSource: {match['source_filename']}")
        print(f"Page: {match['page_number']}")
        print(f"Distance: {match['distance']:.4f}")
        print(match["text"])
