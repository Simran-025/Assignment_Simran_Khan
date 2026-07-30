import streamlit as st
from tempfile import NamedTemporaryFile
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


load_dotenv(
    Path(__file__).parent.parent / ".env"
)

st.set_page_config(
    page_title="Hybrid RAG Assistant",
    page_icon="🤖",
    layout="wide"
)


st.sidebar.title("⚙️ Settings")


model = st.sidebar.selectbox(
    "Choose Groq Model",
    [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile"
    ]
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.1
)


st.sidebar.write(
    f"Temperature: {temperature:.1f}"
)


st.sidebar.divider()

top_k = st.sidebar.slider(
    "Documents to retrieve",
    min_value=1,
    max_value=10,
    value=5,
    step=1
)


bm25_weight = st.sidebar.slider(
    "BM25 Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1
)

dense_weight = 1.0 - bm25_weight


st.sidebar.write(
    f"Dense Weight: {dense_weight:.1f}"
)

st.sidebar.divider()

st.sidebar.caption(
    f"Model: {model}"
)

st.sidebar.caption(
    "Retrieval: BM25 + FAISS"
)

st.title("🤖 Hybrid RAG Assistant")

st.write(
    "Upload a PDF and ask questions about its content."
)

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    with st.spinner("Processing PDF..."):

        with NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(
                uploaded_file.getvalue()
            )

            pdf_path = temp_file.name


        loader = PyPDFLoader(pdf_path)

        docs = loader.load()


        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(
            docs
        )


        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )



        vector_db = FAISS.from_documents(
            chunks,
            embeddings
        )


        dense_retriever = vector_db.as_retriever(
            search_kwargs={
                "k": top_k
            }
        )

        bm25_retriever = BM25Retriever.from_documents(
            chunks
        )

        bm25_retriever.k = top_k



        hybrid_retriever = EnsembleRetriever(
            retrievers=[
                bm25_retriever,
                dense_retriever
            ],
            weights=[
                bm25_weight,
                dense_weight
            ]
        )


    st.success(
        f"PDF loaded successfully! "
        f"{len(docs)} pages found."
    )

    question = st.text_input(
        "Ask anything about your document:"
    )


    if st.button(
        "Ask",
        type="primary"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching the document..."
            ):

      
                results = hybrid_retriever.invoke(
                    question
                )

                context = "\n\n".join(
                    doc.page_content
                    for doc in results
                )

                prompt = ChatPromptTemplate.from_template(
                    """
You are an awesome AI assistant.

Answer the question using pdf uploaded.

If the answer can't be found in the context,
say:

"I could not find the answer in the document."

Context: {context}

Question: {question}

Answer:
"""
                )

                final_prompt = prompt.invoke(
                    {
                        "context": context,
                        "question": question
                    }
                )

                llm = ChatGroq(
                    model=model,
                    temperature=temperature
                )

                response = llm.invoke(
                    final_prompt
                )

            st.subheader("Answer")

            st.write(
                response.content
            )