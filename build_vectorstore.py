from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def read_pdf(file_path):
    text = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def split_text_to_documents(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    texts = splitter.split_text(text)
    documents = splitter.create_documents(texts)
    return documents

def create_faiss_vectorstore(pdf_path, save_dir):
    print(f"Reading PDF from {pdf_path} ...")
    doc_text = read_pdf(pdf_path)

    print("Splitting document into chunks...")
    documents = split_text_to_documents(doc_text)

    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    print("Building FAISS vector store...")
    vectorstore = FAISS.from_documents(documents, embeddings)

    os.makedirs(save_dir, exist_ok=True)
    vectorstore.save_local(save_dir)
    print(f"Vector store saved at {save_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build FAISS vector store from PDF knowledge base")
    parser.add_argument("--pdf", type=str, required=True, help="Path to PDF knowledge base file")
    parser.add_argument("--output", type=str, required=True, help="Directory to save FAISS vector store")

    args = parser.parse_args()
    create_faiss_vectorstore(args.pdf, args.output)
