import streamlit as st
import os
import tempfile

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_classic.chains import ConversationalRetrievalChain

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")

llm = ChatGroq(model="llama-3.1-8b-instant",temperature=0.3)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.header("Chat with your PDF (RAG + Memory)")

st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload your PDF", type=["pdf"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if uploaded_file is not None and st.session_state.vectorstore is None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name

    st.sidebar.success("PDF uploaded successfully!")

    # Load PDF
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    if not docs:
        st.error("No content found in PDF")
        st.stop()

    st.write(f"Total pages loaded: {len(docs)}")
    st.write("Sample content:", docs[0].page_content[:300])

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    st.write(f"Total Chunks Created: {len(chunks)}")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = FAISS.from_documents(chunks, embeddings)
    
    st.session_state.vectorstore = vectordb

    st.sidebar.success("Embedding done successfully!")

    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectordb.as_retriever())

    st.session_state.qa_chain = qa_chain

if st.session_state.qa_chain is not None:

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])
            
    query = st.chat_input(" Ask something from your PDF...")
    if query:
        # Store user message
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)
     
        with st.spinner("Thinking..."):
            result = st.session_state.qa_chain.invoke({
                "question": query,
                "chat_history": [
                    (msg["content"], "") if msg["role"] == "user"
                    else ("", msg["content"])
                    for msg in st.session_state.chat_history
                ]
            })

            response = result["answer"]
            
        with st.chat_message("assistant"):
            st.write(response)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": response}
        )

else:
    st.info("👈 Please upload a PDF to start chatting.")


