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


def fixed_size_chunk(
    document: Document,
    chunk_size: int,
) -> list[Document]:
    """Split a document into fixed-size chunks."""

    chunks: list[Document] = []

    for chunk_id, start in enumerate(
        range(0, len(document.page_content), chunk_size)
    ):

        end = min(start + chunk_size, len(document.page_content))

        chunk_text = document.page_content[start:end]

        metadata = document.metadata.copy()
        metadata["chunk_id"] = chunk_id
        metadata["chunk_start"] = start
        metadata["chunk_end"] = end

        chunks.append(
            Document(
                page_content=chunk_text,
                metadata=metadata,
            )
        )

    return chunks


def main():
    folder = "sample"
    chunk_size = 10

    documents = load_documents(folder)

    print(f"\nLoaded {len(documents)} documents\n")

    for i, document in enumerate(documents, start=1):

        print(f"Document {i}")
        print(f"File       : {document.metadata['file_name']}")
        print(f"Language   : {document.metadata['language']}")
        print(f"Path       : {document.metadata['relative_path']}")
        print(f"Characters : {document.metadata['char_count']}")
        print("-" * 50)

        chunks = fixed_size_chunk(document, chunk_size)

        print(f"Created {len(chunks)} chunks\n")

        for chunk in chunks:
            print(
                f"Chunk {chunk.metadata['chunk_id']} "
                f"({chunk.metadata['chunk_start']}:{chunk.metadata['chunk_end']})"
            )
            print(repr(chunk.page_content))
            print()

        print("=" * 60)


if __name__ == "__main__":
    main()