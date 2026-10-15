import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# On Streamlit Community Cloud there is no .env file -- secrets are entered
# in the app's dashboard and read via st.secrets. Copy them into the normal
# environment so the rest of this file (and ingest.py) doesn't need to care
# whether it's running locally or in the cloud.
for _key in ("GOOGLE_API_KEY", "GEMINI_MODEL", "DOCS_FOLDER"):
    if _key in st.secrets:
        os.environ[_key] = str(st.secrets[_key])

DB_FOLDER = "insti_buddy_db"

# First visitor to a fresh deployment triggers ingestion since the database
# folder is not committed to git (ingest.py reads DOCS_FOLDER itself).
if not Path(DB_FOLDER).exists():
    with st.spinner("Setting up the knowledge base for the first time (this can take a minute)..."):
        import ingest
        ingest.main()
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # MUST match ingest.py
TOP_K = 3

PROMPT = ChatPromptTemplate.from_template("""
You are Insti Buddy, a helpful assistant for IIT Madras students.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say:
"I could not find that information in the campus documents."
Do NOT make up information.

Context:
{context}

Question: {question}

Answer:""")


@st.cache_resource  # load once, reuse across messages instead of reloading every time
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma(persist_directory=DB_FOLDER, embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})


@st.cache_resource
def load_chain():
    llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
    return PROMPT | llm | StrOutputParser()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def source_list(docs):
    seen = []
    for doc in docs:
        name = Path(doc.metadata.get("source", "unknown")).name
        page = doc.metadata.get("page")
        label = f"{name}, page {page + 1}" if isinstance(page, int) else name
        if label not in seen:
            seen.append(label)
    return seen


def answer_question(question):
    docs = load_retriever().invoke(question)
    if not docs:
        return "I could not find that information in the campus documents.", []
    answer = load_chain().invoke({"context": format_docs(docs), "question": question})
    return answer, source_list(docs)


# ---------- The chat window ----------
st.set_page_config(page_title="Insti Buddy", page_icon="🤖")
st.title("Insti Buddy — IIT Madras Campus Assistant")
st.write("Ask me anything about campus life!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write("•", s)

if user_input := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching the campus documents..."):
            answer, sources = answer_question(user_input)
        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.write("•", s)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})