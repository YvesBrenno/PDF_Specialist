import streamlit as st
import os
import time

from src.embeddings import load_embeddings_model
from src.indexing import load_faiss_index
from src.model import TinyLlamaModel
from src.retriever import DocumentRetriever

st.set_page_config(page_title="PDF Specialist", page_icon="📄", layout="wide")
st.title("📄 PDF Specialist")

uploaded_file = st.file_uploader("Envie um documento PDF", type="pdf")

@st.cache_resource(show_spinner=False)
def initialize_components(pdf_path: str):
    with st.spinner("Carregando componentes necessários..."):
        embeddings = load_embeddings_model()
        faiss_index = load_faiss_index(embeddings, pdf_path)
        retriever = DocumentRetriever(faiss_index)
        model = TinyLlamaModel()
        model.load_model()
        return embeddings, faiss_index, retriever, model

if uploaded_file:
    pdf_path = os.path.join("data", uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    embeddings, faiss_index, retriever, model = initialize_components(pdf_path)

    st.subheader("Faça uma pergunta sobre o conteúdo do PDF")
    query = st.text_input("Sua pergunta:", placeholder="Ex: O que o documento diz sobre segurança?")
    num_docs = st.sidebar.slider("Número de trechos a recuperar", min_value=1, max_value=10, value=3)
    max_length = st.sidebar.slider("Comprimento máximo da resposta", min_value=100, max_value=1000, value=300)
    show_docs = st.sidebar.checkbox("Mostrar documentos recuperados", value=False)

    if st.button("Obter Resposta") and query:
        with st.spinner("Processando sua pergunta..."):
            relevant_docs = retriever.retrieve_documents(query, top_k=num_docs)

            if show_docs:
                st.subheader("Trechos relevantes do documento:")
                for i, doc in enumerate(relevant_docs):
                    st.info(f"Trecho {i+1}:\n{doc}")

            start_time = time.time()
            response = model.generate_response(query, relevant_docs, max_length=max_length)
            end_time = time.time()

            st.subheader("Resposta:")
            st.write(response)
            st.caption(f"Tempo de geração: {end_time - start_time:.2f} segundos")

    with st.expander("Sobre este projeto"):
        st.markdown("""
        Este projeto utiliza:
        - **LangChain** para processamento e chunking
        - **TinyLlama** para geração de respostas
        - **FAISS** para busca vetorial
        - **Streamlit** como interface
        """)

    st.markdown("---")
    st.caption("PDF Specialist - Brenno Yves")
