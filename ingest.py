import os
from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DOCS_FOLDER = os.getenv("DOCS_FOLDER", "campus_docs")  # "public_docs" on the deployed demo
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DB_FOLDER = "insti_buddy_db"


def load_pdfs(folder):
    documents = []
    for pdf_file in sorted(Path(folder).glob("*.pdf")):
        reader = PdfReader(pdf_file)
        for page_number, page in enumerate(reader.pages):
            documents.append(Document(
                page_content=page.extract_text() or "",
                metadata={"source": str(pdf_file), "page": page_number},
            ))
    return documents
def split_into_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)


def create_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def store_in_database(chunks, embeddings):
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_FOLDER,
    )


def main():
    documents = load_pdfs(DOCS_FOLDER)
    print(f"Loaded {len(documents)} pages")

    chunks = split_into_chunks(documents)
    print(f"Split into {len(chunks)} chunks")

    embeddings = create_embeddings()
    store_in_database(chunks, embeddings)
    print(f"Saved {len(chunks)} chunks to '{DB_FOLDER}'. Done!")


if __name__ == "__main__":
    main()
