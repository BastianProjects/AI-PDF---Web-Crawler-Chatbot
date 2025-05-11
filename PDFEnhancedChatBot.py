
import streamlit as st
from PDFhandling import load_pdf, upload_pdf, pdfs_directory, figures_directory
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import os
import shutil
import gc

# Ensure required binaries are in the PATH
os.environ["PATH"] += os.pathsep + r"D:\Release-24.08.0-0\poppler-24.08.0\Library\bin" + os.pathsep + r"C:\Program Files\Tesseract-OCR"

# -- Settings --
CHROMA_DB_DIR = "chroma_DB"
COLLECTION_NAME = "everything"
EMBED_MODEL = "llama3.2"
LLM_MODEL = "gemma3"

# -- Templates --
template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context and chat history to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Chat History:
{history}

Context:
{context}

Question: {question}
Answer:"""

# -- Initialize Models --
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
llm = OllamaLLM(model=LLM_MODEL)

@st.cache_resource
def get_vector_store():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

# -- Memory Initialization --
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def load_page(url):
    loader = SeleniumURLLoader(urls=[url])
    documents = loader.load()
    if not documents:
        st.error(f"Error: No content loaded from {url}")
    else:
        st.success(f"Successfully loaded {len(documents)} documents from {url}.")
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    if not chunks:
        st.error("Error: No valid chunks created.")
    else:
        st.success(f"Successfully created {len(chunks)} chunks.")
    return chunks

def index_docs(docs, source_label):
    store = get_vector_store()
    wrapped = [
        Document(page_content=doc.page_content, metadata={"source_url": source_label})
        for doc in docs
    ]
    if not wrapped:
        st.error("Error: No documents to index.")
    else:
        st.success(f"Indexing {len(wrapped)} documents from {source_label}.")
    store.add_documents(wrapped)
    store.persist()

def retrieve_docs(query, filter_source=None):
    store = get_vector_store()
    if filter_source and filter_source != "All":
        return store.similarity_search(query, k=4, filter={"source_url": filter_source})
    
    all_metadatas = store.get()["metadatas"]
    sources = set(meta["source_url"] for meta in all_metadatas if "source_url" in meta)
    results = []
    for src in sources:
        results += store.similarity_search(query, k=2, filter={"source_url": src})
    return results

def answer_question(question, context):
    history = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.chat_history[-6:]]
    )
    full_prompt = ChatPromptTemplate.from_template(template)
    chain = full_prompt | llm
    return chain.invoke({"question": question, "context": context, "history": history})

def summarize_pdf_text(text: str):
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following PDF content in a paragraph:\n{context}"
    )
    chain = prompt | llm
    return chain.invoke({"context": text[:3000]})

# -- Streamlit UI --
st.title("🔎 AI PDF & Web Crawler Chatbot")

if st.sidebar.button("🧹 Clear Vector DB"):
    try:
        if "vector_store" in st.session_state:
            del st.session_state["vector_store"]
        get_vector_store.clear()
    except Exception as e:
        st.sidebar.error(f"Error clearing vector store: {e}")
    gc.collect()
    if os.path.exists(CHROMA_DB_DIR):
        try:
            shutil.rmtree(CHROMA_DB_DIR)
            st.sidebar.success("✅ Vector store cleared.")
        except PermissionError:
            st.sidebar.error("❌ Could not delete vector store. Try restarting the app.")

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.sidebar.success("Chat history cleared.")

store = get_vector_store()
try:
    all_sources_raw = store.get()["metadatas"]
    all_sources = sorted(set(meta.get("source_url") for meta in all_sources_raw if meta and meta.get("source_url")))
    st.sidebar.markdown("### 📚 Indexed Sources:")
    for u in all_sources:
        label = f"PDF: {u}" if u.endswith(".pdf") else f"URL: {u}"
        st.sidebar.write(f"- {label}")
except Exception as e:
    all_sources = []
    st.sidebar.error("Could not list sources.")

url = st.text_input("Enter a Web Page URL to index:")
if url:
    with st.spinner("🔄 Loading and indexing page..."):
        documents = load_page(url)
        chunks = split_text(documents)
        index_docs(chunks, url)
    st.success("✅ Page indexed!")

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        upload_pdf(uploaded_file)
        text = load_pdf(pdfs_directory + uploaded_file.name)
        summary = summarize_pdf_text(text)
        st.chat_message("📄 PDF Summary").write(summary)
        documents = [Document(page_content=text)]
        chunked_texts = split_text(documents)
        index_docs(chunked_texts, uploaded_file.name)

selected_source = st.sidebar.selectbox("Filter answers by source (or 'All'):", ["All"] + all_sources)
question = st.chat_input("Ask a question...")

if question:
    st.chat_message("user").write(question)
    with st.spinner("🤖 Thinking..."):
        results = retrieve_docs(question, selected_source)
        context = "\n\n".join([doc.page_content for doc in results])
        answer = answer_question(question, context)
    st.chat_message("assistant").write(answer)
    st.session_state.chat_history.append({"role": "user", "content": question})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    with st.expander("📄 Retrieved Chunks"):
        for i, doc in enumerate(results, 1):
            st.markdown(
                f"**Chunk {i} (from {doc.metadata.get('source_url')}):**\n\n{doc.page_content[:500]}..."
            )
