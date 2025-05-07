from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings

def load_embeddings_model():
    print("Carregando modelo de embeddings...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    print(f"Modelo de embeddings {model_name} carregado com sucesso!")
    return embeddings
