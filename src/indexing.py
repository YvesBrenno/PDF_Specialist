import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS

def create_faiss_index(embeddings, pdf_path: str, index_dir="data"):
    print("Carregando o documento...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print("Dividindo o documento em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    texts = text_splitter.split_documents(documents)

    print(f"Documento dividido em {len(texts)} chunks.")

    print("Criando índice FAISS...")
    faiss_index = FAISS.from_documents(texts, embeddings)

    index_path = os.path.join(index_dir, "generic_index")
    faiss_index.save_local(index_path)
    print(f"Índice FAISS salvo em {index_path}")

    return faiss_index

def load_faiss_index(embeddings, pdf_path: str, index_dir="data"):
    index_path = os.path.join(index_dir, "generic_index")

    if os.path.exists(index_path):
        print(f"Carregando índice FAISS existente de {index_path}...")
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Índice FAISS não encontrado. Criando novo índice...")
        return create_faiss_index(embeddings, pdf_path, index_dir)