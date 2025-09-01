# Streamlit RAG chatbot using PDF + TF IDF (scikit-learn) + chromaDB + Langchain utilities
import os
import joblib
import uuid
import streamlit as st

from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ------------Config----------
PERSIST_DIR = "chroma_tfidf_store"
VECTORIZER_PATH = os.path.join(PERSIST_DIR,"tfidf_vectorizer.joblib")

#------------- Embeddings wrapper (scikit-learn TF-IDF) ------------
class SklearnTfidfEmbeddings(Embeddings):
    """
    A minimal embeddings interface for LangChain that uses scikit-learn TF-IDF.
    We fit on the corpus(documents) once, save the vectorizer, and reuse it for queries.
    """

    def __init__(self, vectorizer: Optional[TfidfVectorizer]=None):
        self.vectorizer=vectorizer or TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=100_000,
            ngram_range=(1,2)
        )
        self.fitted=False

    def fit(self,texts:List[str]):
        self.vectorizer.fit(texts)
        self.fitted=True

    def embed_documents(self,texts:List[str]) -> List[List[float]]:
        # If not fitted, fit on incoming documents
        if not self.fitted:
            self.fit(texts)
        X=self.vectorizer.transform(texts)
        return X.toarray().tolist()

    def embed_query(self,text:str) -> List[float]:
        if not self.fitted:
            raise ValueError("TF-IDF vectorizer is not fitted yet. Index documents first.")
        x = self.vectorizer.transform([text])
        return x.toarray()[0].tolist()


# ---------------Helpers----------------
def load_pdf_to_docs(file_path:str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    docs=loader.load()
    return docs

def chunk_docs(docs: List[Document], chunk_size=1000, chunk_overlap=200) -> List[Document]:
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n"," ",""]
    )  
    return splitter.split_documents(docs)

def ensure_dirs():
    os.makedirs(PERSIST_DIR,exist_ok=True)

def create_or_load_vectorizer() -> SklearnTfidfEmbeddings:
    ensure_dirs()
    if os.path.exists(VECTORIZER_PATH):
        vec: TfidfVectorizer = joblib.load(VECTORIZER_PATH)
        emb = SklearnTfidfEmbeddings(vec)
        emb.fitted=True
        return emb
    return SklearnTfidfEmbeddings()

def persist_vectorizer(emb:SklearnTfidfEmbeddings):
    ensure_dirs()
    joblib.dump(emb.vectorizer, VECTORIZER_PATH)

def build_or_update_index(chunks: List[Document], emb: SklearnTfidfEmbeddings) -> Chroma:
    """
    If a Chroma index exists, add to it. Otherwise create it.
    """    
    ensure_dirs()
    if os.listdir(PERSIST_DIR):
        # Chroma exists, load it
        vectordb = Chroma(
            embedding_function=emb,
            persist_directory=PERSIST_DIR
        )
        vectordb.add_documents(chunks)
        # vectordb.persist()
        return vectordb
    
    else:
        vectordb = Chroma.from_documents(
            document=chunks,
            embedding=emb,
            persist_directory=PERSIST_DIR
        )
        # vectordb.persist()
        return vectordb

def reset_index():
    if os.path.exists(PERSIST_DIR):
        # Dangerous delete; Chroma stores multiple files-clear the folder
        for root,dirs,files in os.walk(PERSIST_DIR, topdown=False):
            for name in files:
                os.remove(os.path.join(root,name))
            for name in dirs:
                os.rmdir(os.path.join(root,name))
        try:
            os.rmdir(PERSIST_DIR)
        except OSError:
            pass
    if os.path.exists(VECTORIZER_PATH):
        try:
            os.remove(VECTORIZER_PATH)
        except OSError:
            pass

# ------------------STREAMLIT UI ----------------------

st.set_page_config(page_title="PDF RAG CHATBOT(TF-IDF + Chroma)", layout="wide")
st.title("PDF RAG Chatbot - TF-IDF + ChromaDB")
st.caption("Local PDF -> Chunk -> TF-IDF (scikit-learn) -> ChromaDB -> Retrieve top chunks -> Chat")

with st.sidebar:
    st.header("Settings")
    chunk_size=st.number_input("Chunk Size", min_value=200, max_value=4000, value=1000, step=100)
    chunk_overlap=st.number_input("Chunk overlap", min_value=0, max_value=1000, value=200, step=50)
    top_k = st.slider("Top K results", min_value=1, max_value=10, value=4)
    if st.button("Reset Index(clear DB)", type="secondary"):
        reset_index()
        st.success("Index cleared.")
    st.markdown("----")
    st.write("**Persistence folder:**")
    st.code(PERSIST_DIR, language="bash")

# Session state
if "message" not in st.session_state:
    st.session_state.messages =[]

# File Uploader
uploaded_files = st.file_uploader(
    "Upload one or more PDFs to index", type=["pdf"], accept_multiple_files=True
)

col_left,col_right = st.columns([3,2],gap="large")

with col_left:
    st.subheader("1) Index PDFs")
    if uploaded_files:
        temp_paths=[]
        for f in uploaded_files:
            # Save uploaded file to a temp path
            temp_name = f"{uuid.uuid4()}.pdf"
            temp_path=os.path.join("tmp_uploads",temp_name)
            os.makedirs("tmp_uploads",exist_ok=True)
            with open(temp_path, "wb") as out:
                out.write(f.read())
            temp_paths.append(temp_path)

        if st.button("Build/Update Index"):
            all_docs: List[Document]=[]
            for p in temp_paths:
                all_docs.extend(load_pdf_to_docs(p))
            st.info(f"Loaded {len(all_docs)} pages. Chunking....")
            chunks=chunk_docs(all_docs,chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            st.success(f"Created {len(chunks)} chunks..")
            
            # Create / load TF-IDF embeddings
            emb = create_or_load_vectorizer()

            # Build or update Chroma index
            vectordb = build_or_update_index(chunks,emb)

            # Persist vectorizer (critical so queries work in later sessions)
            persist_vectorizer(emb)

            st.success("Index built/updated and persisted")

with col_right:
    st.subheader("2) Test the Retriever")
    test_query = st.text_input("Try a quick search (no chat):")
    if st.button("Retrieve"):
        try:
            emb=create_or_load_vectorizer()
            vectordb=Chroma(embedding_function=emb, persist_directory=PERSIST_DIR)
            results=vectordb.similarity_search(test_query, k=top_k)
            for i,r in enumerate(results,1):
                with st.expander(f"Result {i}"):
                    st.write(r.page_content)
                    md=r.metadata or {}
                    src=f"Page: {md.get('page','NA')}, Source: {md.get('source','NA')}"
                    st.caption(src)
        except Exception as e:
            st.error(str(e))

st.markdown("---")


#------------------------Chatbot-----------------------
st.subheader("Chat with your PDF")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input=st.chat_input("Ask anything from indexed PDFs")

if user_input:
    # Show user message
    st.session_state.messages.append({"role":"user","content":user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

# Retrieve top chunks
try:
    emb=create_or_load_vectorizer()
    vectordb=Chroma(embedding_function=emb, persist_directory=PERSIST_DIR)
    results=vectordb.similarity_search(user_input,k=top_k)
    # Build a sample answer by stitching the top chunks
    stitched = "\n\n---\n\n".join(
        [r.page_content.strip() for r in results]
    )
    # You can optionally truncate to keep responses compact
    max_chars=2000
    answer = stitched[:max_chars] + (" ..." if len(stitched) > max_chars else "")
    bot_msg=(
        "**Top matches from your PDF(s): **\n\n"
        f"{answer}\n\n"
        "_(TF-IDF retrieval; no generative model used)_"
    )
except Exception as e:
    bot_msg=f"Error: {e}"

# Show assistant message
with st.chat_message("assistant"):
    st.markdown(bot_msg)
st.session_state.messages.append({"role":"assistant","content":bot_msg})    


