from dataclasses import dataclass
import os
from typing import Any 

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


folder = "sample"
documents = []

for filename in sorted(os.listdir(folder)):
    filepath = os.path.join(folder, filename)

    if not os.path.isfile(filepath):
        continue

    _, ext = os.path.splitext(filename)

    if ext not in EXTENSION_TO_LANGUAGE:
        continue

    with open(filepath, "r", encoding="utf-8") as file:
        text = file.read()

    doc = Document(
        page_content=text,
        metadata={
            "file_name": filename,
            "relative_path": filepath,
            "language": EXTENSION_TO_LANGUAGE[ext],
            "char_count": len(text),
        },
    )

    documents.append(doc)

print(f"Loaded {len(documents)} documents\n")

for i, doc in enumerate(documents, start=1):
    print(f"Document {i}")
    print(f"File       : {doc.metadata['file_name']}")
    print(f"Language   : {doc.metadata['language']}")
    print(f"Path       : {doc.metadata['relative_path']}")
    print(f"Characters : {doc.metadata['char_count']}")
    print("-" * 50)