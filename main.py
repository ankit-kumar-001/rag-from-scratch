from dataclasses import dataclass
from typing import Any
import os

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


def main():
    folder = "sample"

    chunk_size = 30

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

        for chunk in chunks:
            print(
                f"Chunk {chunk.metadata['chunk_id']}"
            )
            print(repr(chunk.page_content))
            print()

        print("=" * 60)


if __name__ == "__main__":
    main()