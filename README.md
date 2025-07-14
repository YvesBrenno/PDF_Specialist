# PDF Specialist

Este projeto é um sistema de perguntas e respostas baseado em documentos PDF. Ele usa a técnica de Retrieval Augmented Generation (RAG) para combinar busca semântica e geração de texto com IA. O modelo responde com base no conteúdo do PDF enviado.

## 🔧 Tecnologias utilizadas

- [LangChain](https://www.langchain.com/): processamento e chunking de documentos
- [FAISS](https://github.com/facebookresearch/faiss): indexação vetorial
- [Hugging Face Transformers](https://huggingface.co/): embeddings e modelo TinyLlama
- [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0): modelo leve de linguagem
- [Streamlit](https://streamlit.io/): interface web

---

## 🚀 Como rodar o projeto

### 1. Clone o repositório
```bash
git clone https://github.com/YvesBrenno/PDF_Specialist.git
cd PDF_Specialist
```

### 2. Crie e ative um ambiente virtual
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate    # Windows
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Execute o app com o Streamlit
```bash
streamlit run app.py
```

---

## Como usar
1. Suba um arquivo PDF usando o botão "Envie um documento PDF"
2. Faça perguntas no campo de texto baseado no conteúdo do documento
3. Veja a resposta gerada com base no conteúdo mais relevante

Você também pode controlar o número de trechos recuperados e o tamanho da resposta.

---

## Observações
- Funciona melhor com PDFs que tenham texto (não imagens)
- Documentos muito grandes podem consumir mais RAM e CPU
- As respostas são baseadas **apenas** no conteúdo do PDF enviado

---

