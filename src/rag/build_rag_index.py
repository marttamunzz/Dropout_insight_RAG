from pathlib import Path
from typing import Any

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parents[2]

PDF_DIR = BASE_DIR / "rag_papers"
DB_DIR = str(BASE_DIR / "rag_db")

COLLECTION_NAME = "dropout_literature"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def read_pdf(pdf_path: Path) -> list[dict[str, Any]]:
    reader = PdfReader(str(pdf_path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "source": pdf_path.name,
                "page": page_number,
                "text": text.strip(),
            })

    return pages


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

        if start < 0:
            start = 0

        if start >= len(text):
            break

    return chunks


def build_index() -> None:
    if not PDF_DIR.exists():
        raise FileNotFoundError(
            f"PDF folder not found: {PDF_DIR}. "
            "Create it and add your literature PDFs."
        )

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {PDF_DIR}."
        )

    print(f"Found {len(pdf_files)} PDF files.")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    client = chromadb.PersistentClient(path=DB_DIR)

    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Previous collection '{COLLECTION_NAME}' deleted.")
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    ids = []
    documents = []
    metadatas = []

    chunk_counter = 0

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")

        pages = read_pdf(pdf_path)

        for page in pages:
            chunks = split_text(
                text=page["text"],
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
            )

            for local_chunk_index, chunk in enumerate(chunks):
                chunk_id = f"{pdf_path.stem}_p{page['page']}_c{local_chunk_index}"

                ids.append(chunk_id)
                documents.append(chunk)
                metadatas.append({
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_index": local_chunk_index,
                })

                chunk_counter += 1

    print(f"Creating embeddings for {len(documents)} chunks...")

    embeddings = embedding_model.encode(
        documents,
        show_progress_bar=True,
    ).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print("RAG index created successfully.")
    print(f"Database path: {DB_DIR}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Chunks stored: {chunk_counter}")


if __name__ == "__main__":
    build_index()