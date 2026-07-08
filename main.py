from dataclasses import dataclass
from typing import Any
from google import genai
from dotenv import load_dotenv
import chromadb
import os

load_dotenv()
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".cpp": "cpp",
    ".h": "cpp-header",
    ".hpp": "cpp-header",
    ".java": "java",
    ".js": "javascript",
    ".ts": "typescript",
    ".md": "markdown",
    ".txt": "text",
}

SEPARATORS = [
    "\n\n",
    "\n",
    " ",
]


api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY environment variable not found."
    )


client = genai.Client(
    api_key=api_key
)


@dataclass
class Document:
    page_content: str
    metadata: dict[str, Any]


def load_documents(folder: str) -> list[Document]:
    """Read all supported files from a folder and return Document objects."""

    documents: list[Document] = []

    for filename in sorted(os.listdir(folder)):
        filepath = os.path.join(folder, filename)

        if not os.path.isfile(filepath):
            continue

        _, ext = os.path.splitext(filename)

        if ext not in EXTENSION_TO_LANGUAGE:
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as file:
                text = file.read()

        except Exception as e:
            print(f"Failed to read {filepath}: {e}")
            continue

        doc = Document(
            page_content=text,
            metadata={
                "file_name": filename,
                "relative_path": os.path.relpath(filepath, folder),
                "language": EXTENSION_TO_LANGUAGE[ext],
                "char_count": len(text),
            },
        )

        documents.append(doc)

    return documents


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for multiple texts in a single API call.
    """

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
    )

    return [
        embedding.values
        for embedding in response.embeddings
    ]


def batched(items: list, batch_size: int):
    """Yield successive sub-lists of at most batch_size items."""
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def embed_documents(chunks: list[Document], batch_size: int = 50) -> list[list[float]]:
    """
    Embed a list of chunk Documents, sending sub-batches of at most
    batch_size texts per API call to stay under rate/size limits.
    """

    all_embeddings: list[list[float]] = []

    for sub_batch in batched(chunks, batch_size):
        texts = [chunk.page_content for chunk in sub_batch]
        embeddings = embed_batch(texts)
        all_embeddings.extend(embeddings)

    return all_embeddings


def fixed_size_chunker(
    document: Document,
    chunk_size: int,
    overlap: int = 0,
) -> list[Document]:
    """Split a document into fixed-size chunks with optional overlap."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[Document] = []

    step = chunk_size - overlap

    for chunk_id, start in enumerate(
        range(0, len(document.page_content), step)
    ):
        end = min(start + chunk_size, len(document.page_content))

        chunk_text = document.page_content[start:end]

        if not chunk_text:
            continue

        metadata = {
            **document.metadata,
            "chunk_id": chunk_id,
            "chunk_start": start,
            "chunk_end": end,
        }

        chunks.append(
            Document(
                page_content=chunk_text,
                metadata=metadata,
            )
        )

    return chunks


def recursive_chunk(
    document: Document,
    chunk_size: int,
    separator_index: int = 0,
) -> list[Document]:

    if separator_index >= len(SEPARATORS):
        return fixed_size_chunker(document, chunk_size)

    separator = SEPARATORS[separator_index]

    pieces = document.page_content.split(separator)

    chunks: list[Document] = []

    current_chunk = ""
    chunk_id = 0

    for piece in pieces:

        if not piece.strip():
            continue

        # Piece itself is too large -> recurse
        if len(piece) > chunk_size:

            # Save current merged chunk before recursion
            if current_chunk:

                metadata = {
                    **document.metadata,
                    "chunk_id": chunk_id,
                }

                chunks.append(
                    Document(
                        page_content=current_chunk,
                        metadata=metadata,
                    )
                )

                chunk_id += 1
                current_chunk = ""

            child_document = Document(
                page_content=piece,
                metadata=document.metadata.copy(),
            )

            child_chunks = recursive_chunk(
                child_document,
                chunk_size,
                separator_index + 1,
            )

            for chunk in child_chunks:
                chunk.metadata["chunk_id"] = chunk_id
                chunks.append(chunk)
                chunk_id += 1

            continue

        # Try to merge with current chunk
        if not current_chunk:
            candidate = piece
        else:
            candidate = current_chunk + separator + piece

        if len(candidate) <= chunk_size:
            current_chunk = candidate

        else:

            metadata = {
                **document.metadata,
                "chunk_id": chunk_id,
            }

            chunks.append(
                Document(
                    page_content=current_chunk,
                    metadata=metadata,
                )
            )

            chunk_id += 1

            current_chunk = piece

    # Save remaining chunk
    if current_chunk:

        metadata = {
            **document.metadata,
            "chunk_id": chunk_id,
        }

        chunks.append(
            Document(
                page_content=current_chunk,
                metadata=metadata,
            )
        )

    return chunks


def store_chunks(
    collection,
    chunks: list[Document],
    embeddings: list[list[float]],
):
    """
    Store chunk documents and their embeddings in a ChromaDB collection
    using a single batched insertion.
    """

    ids = []
    documents = []
    metadatas = []
    vectors = []

    for chunk, embedding in zip(chunks, embeddings):

        chunk_id = f"{chunk.metadata['file_name']}_{chunk.metadata['chunk_id']}"

        ids.append(chunk_id)
        documents.append(chunk.page_content)
        metadatas.append(chunk.metadata)
        vectors.append(embedding)

    collection.add(
        ids=ids,
        embeddings=vectors,
        documents=documents,
        metadatas=metadatas,
    )


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string using the exact same embedding model
    used for document chunks. Reuses embed_batch so query and document
    embeddings are guaranteed to come from the same call path/model.
    """

    embeddings = embed_batch([query])
    return embeddings[0]


def retrieve(
    query: str,
    collection,
    top_k: int = 3,
) -> list[Document]:
    """
    Embed a query, run a semantic search against the ChromaDB collection,
    and convert the results back into our own Document objects.
    """

    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    # results is keyed by field name -> list of lists (one inner list per query)
    documents_texts = results["documents"][0]
    metadatas = results["metadatas"][0]

    retrieved_documents: list[Document] = []

    for text, metadata in zip(documents_texts, metadatas):
        retrieved_documents.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

    return retrieved_documents


def generate_answer(
    query: str,
    documents: list[Document],
) -> str:
    """
    Given a user query and the retrieved Document chunks, ask Gemini to
    answer the query using only that context.
    """

    if not documents:
        return "No relevant documents found."

    context = "\n\n-----\n\n".join(
        document.page_content for document in documents
    )

    prompt = f"""You are an AI assistant that answers questions about a GitHub repository.

Use ONLY the provided context.

If the answer is not present in the context, say so.

Context:
{context}

Question:
{query}

Answer:"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text


def test_retrieval(collection):
    """Run a handful of sample queries and print what comes back."""

    test_queries = [
        "Where is JWT verification implemented?",
        "Show me authentication code",
        "Which file contains login logic?",
        "How is the token verified?",
    ]

    for query in test_queries:
        print(f"Query: {query}")
        print("-" * 50)

        results = retrieve(query, collection, top_k=3)

        for doc in results:
            print(f"File     : {doc.metadata.get('file_name')}")
            print(f"Chunk id : {doc.metadata.get('chunk_id')}")
            print(f"Text     : {repr(doc.page_content)}")
            print()

        print("=" * 60)


def main():
    folder = "sample"

    chunk_size = 500
    embedding_batch_size = 50  # chunks per API call

    # Set up ChromaDB persistent client and collection
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    collection = chroma_client.get_or_create_collection(
        name="repository",
    )

    documents = load_documents(folder)

    print(f"\nLoaded {len(documents)} documents\n")

    for i, document in enumerate(documents, start=1):

        print(f"Document {i}")
        print(f"File       : {document.metadata['file_name']}")
        print(f"Language   : {document.metadata['language']}")
        print(f"Path       : {document.metadata['relative_path']}")
        print(f"Characters : {document.metadata['char_count']}")
        print("-" * 50)

        chunks = recursive_chunk(
            document=document,
            chunk_size=chunk_size,
        )

        print(f"Created {len(chunks)} chunks\n")

        if not chunks:
            print("=" * 60)
            continue

        embeddings = embed_documents(chunks, batch_size=embedding_batch_size)

        store_chunks(
            collection,
            chunks,
            embeddings,
        )

        for chunk, embedding in zip(chunks, embeddings):
            print(f"Chunk {chunk.metadata['chunk_id']}")
            print(f"Embedding Dimension : {len(embedding)}")
            print(f"First 5 Values      : {embedding[:5]}")
            print(f"Chunk Text          : {repr(chunk.page_content)}")
            print()

        print("=" * 60)

    print(f"\nStored {collection.count()} total chunk embeddings in ChromaDB\n")

    # Test retrieval now that ingestion is done
    print("\nTesting retrieval...\n")
    test_retrieval(collection)

    # Test the full RAG flow: retrieve() -> generate_answer()
    print("\nTesting full RAG flow...\n")

    query = "Where is JWT verification implemented?"

    retrieved_documents = retrieve(query, collection, top_k=3)
    answer = generate_answer(query, retrieved_documents)

    print(f"Query  : {query}")
    print(f"Answer : {answer}")


if __name__ == "__main__":
    main()