import streamlit as st
import fitz
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="BookMind AI",
    page_icon="📚",
    layout="centered"
)


# =========================
# LOAD CSS
# =========================

with open("style.css", "r", encoding="utf-8") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )


# =========================
# GROQ API
# =========================

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)


# =========================
# EMBEDDING MODEL
# =========================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# =========================
# PROCESS PDF
# =========================

def process_pdf(uploaded_file):

    pdf_bytes = uploaded_file.read()

    pdf = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    documents = []

    for page_number, page in enumerate(
        pdf,
        start=1
    ):

        text = page.get_text().strip()

        if text:

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": page_number,
                        "source": uploaded_file.name
                    }
                )
            )

    pdf.close()


    # Chunking

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(
        documents
    )


    # Embeddings

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=False
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")


    # FAISS

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)


    return index, chunks


# =========================
# SEARCH PDF
# =========================

def search_pdf(
    query,
    index,
    chunks,
    k=5
):

    query_embedding = embedding_model.encode(
        [query]
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")


    distances, indices = index.search(
        query_embedding,
        k
    )


    results = []

    for idx, distance in zip(
        indices[0],
        distances[0]
    ):

        if idx == -1:
            continue

        results.append({
            "text": chunks[idx].page_content,
            "page": chunks[idx].metadata["page"],
            "distance": float(distance)
        })


    return results


# =========================
# ASK BOOKMIND
# =========================

def ask_bookmind(
    question,
    index,
    chunks
):

    results = search_pdf(
        question,
        index,
        chunks,
        k=5
    )


    context = "\n\n".join(
        f"[Page {result['page']}]\n{result['text']}"
        for result in results
    )


    prompt = f"""
You are BookMind AI, a helpful PDF assistant.

Answer the user's question using ONLY the provided PDF context.

If the answer is not available in the context, say:

"I couldn't find this information in the uploaded PDF."

Give a clear, natural and concise answer.

Do NOT mention page numbers.
Do NOT include source references.
Do NOT say "according to page".

PDF CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""


    response = client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2
    )


    return response.choices[0].message.content


# =========================
# HEADER
# =========================

st.title("📚 BookMind AI")

st.markdown(
    '<p class="subtitle">Your intelligent PDF assistant</p>',
    unsafe_allow_html=True
)

st.divider()


# =========================
# PDF UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "📄 Upload your PDF",
    type=["pdf"]
)


# =========================
# PROCESS PDF
# =========================

if uploaded_file is not None:

    if (
        "uploaded_file_name" not in st.session_state
        or
        st.session_state.uploaded_file_name
        != uploaded_file.name
    ):

        with st.spinner(
            "📖 Reading your PDF..."
        ):

            index, chunks = process_pdf(
                uploaded_file
            )


        st.session_state.index = index

        st.session_state.chunks = chunks

        st.session_state.uploaded_file_name = (
            uploaded_file.name
        )

        st.session_state.messages = []


    st.success(
        "PDF loaded successfully! ✅"
    )


    st.divider()


    # =========================
    # BOOKMIND
    # =========================

    st.subheader("🤖 BookMind")

    st.write(
        "Ask anything about your PDF"
    )


    # =========================
    # CHAT HISTORY
    # =========================

    if "messages" not in st.session_state:

        st.session_state.messages = []


    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )


    # =========================
    # CHAT INPUT
    # =========================

    question = st.chat_input(
        "Ask a question..."
    )


    if question:

        # User message

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })


        with st.chat_message("user"):

            st.write(question)


        # AI answer

        with st.chat_message("assistant"):

            with st.spinner(
                "Thinking..."
            ):

                answer = ask_bookmind(
                    question,
                    st.session_state.index,
                    st.session_state.chunks
                )


            st.write(answer)


        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })


# =========================
# FOOTER
# =========================

st.divider()

st.markdown(
    '<p class="footer">📚 BookMind AI</p>',
    unsafe_allow_html=True
)