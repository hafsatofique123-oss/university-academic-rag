import os
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="University Academic Assistant",
    page_icon="🎓",
    layout="wide"
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

FAISS_PATH = DATA_DIR / "university_faiss.index"
METADATA_PATH = DATA_DIR / "university_metadata.pkl"
CONFIG_PATH = DATA_DIR / "university_config.json"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #666;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .source-box {
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 10px;
        background-color: #fafafa;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎓 University Academic Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Ask questions about the university's academic documents.
    Answers are generated using the indexed academic knowledge base.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD CONFIGURATION
# ============================================================

@st.cache_data
def load_config():

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# LOAD FAISS INDEX
# ============================================================

@st.cache_resource
def load_faiss_index():

    return faiss.read_index(
        str(FAISS_PATH)
    )


# ============================================================
# LOAD METADATA
# ============================================================

@st.cache_data
def load_metadata():

    with open(
        METADATA_PATH,
        "rb"
    ) as file:

        return pickle.load(file)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model(model_name):

    return SentenceTransformer(
        model_name
    )


# ============================================================
# GET GROQ API KEY
# ============================================================

def get_groq_api_key():

    # Streamlit Cloud / local Streamlit secrets

    if "GROQ_API_KEY" in st.secrets:

        return st.secrets["GROQ_API_KEY"]


    # Optional local environment variable

    api_key = os.environ.get(
        "GROQ_API_KEY"
    )

    if api_key:
        return api_key


    return None


# ============================================================
# LOAD DATABASE
# ============================================================

if not FAISS_PATH.exists():

    st.error(
        "FAISS index not found. "
        "Please make sure the data folder contains "
        "university_faiss.index."
    )

    st.stop()


if not METADATA_PATH.exists():

    st.error(
        "Metadata file not found. "
        "Please make sure the data folder contains "
        "university_metadata.pkl."
    )

    st.stop()


if not CONFIG_PATH.exists():

    st.error(
        "Configuration file not found."
    )

    st.stop()


config = load_config()

faiss_index = load_faiss_index()

metadata = load_metadata()

embedding_model = load_embedding_model(
    config["embedding_model"]
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Retrieval Settings")

    top_k = st.slider(
        "Number of sources to retrieve",
        min_value=2,
        max_value=8,
        value=5,
        step=1
    )

    st.divider()

    st.write("### Knowledge Base")

    st.write(
        f"📚 Documents: "
        f"{config.get('document_count', 'N/A')}"
    )

    st.write(
        f"🧩 Chunks: "
        f"{config.get('chunk_count', len(metadata))}"
    )

    st.write(
        f"🔎 Embedding model: "
        f"{config['embedding_model']}"
    )

    st.write(
        "🗂️ Vector DB: FAISS"
    )

    st.divider()

    st.caption(
        "The original PDF documents are not stored "
        "in this application."
    )


# ============================================================
# RETRIEVAL FUNCTION
# ============================================================

def retrieve_documents(
    query,
    number_of_results
):

    # Create embedding for the student's question.

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )


    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )


    # Search FAISS.

    scores, indices = faiss_index.search(
        query_embedding,
        number_of_results
    )


    results = []


    for score, index in zip(
        scores[0],
        indices[0]
    ):

        # FAISS can return -1 if no result exists.

        if index < 0:
            continue


        document = metadata[index].copy()

        document["similarity_score"] = float(
            score
        )

        results.append(document)


    return results


# ============================================================
# BUILD CONTEXT FOR GROQ
# ============================================================

def build_context(results):

    context_parts = []


    for number, result in enumerate(
        results,
        start=1
    ):

        source = result.get(
            "file_name",
            result.get("source", "Unknown")
        )

        page = result.get(
            "page_number",
            "Unknown"
        )

        chunk = result.get(
            "chunk_number",
            "Unknown"
        )

        text = result.get(
            "text",
            ""
        )


        context_parts.append(
            f"""
SOURCE {number}
File: {source}
Page: {page}
Chunk: {chunk}

Content:
{text}
"""
        )


    return "\n\n".join(
        context_parts
    )


# ============================================================
# GENERATE ANSWER USING GROQ
# ============================================================

def generate_answer(
    question,
    context
):

    api_key = get_groq_api_key()


    if not api_key:

        st.error(
            "GROQ_API_KEY is not configured. "
            "Add it to Streamlit Secrets."
        )

        return None


    client = Groq(
        api_key=api_key
    )


    system_prompt = """
You are a University Student Academic Knowledge Assistant.

Your job is to answer student questions using ONLY
the academic context provided to you.

Rules:

1. Use the provided context as your primary source.
2. Do not invent information.
3. Do not use information that is not supported
   by the retrieved context.
4. If the answer cannot be found in the provided
   context, clearly say that the information was
   not found in the available academic documents.
5. Explain concepts in clear and student-friendly language.
6. Keep the answer focused on the student's question.
7. When possible, mention the relevant source document
   and page number.
8. Do not create fake citations.
9. Do not claim that a source says something unless
   that information actually appears in the context.
"""


    user_prompt = f"""
Answer the following student question.

STUDENT QUESTION:
{question}


ACADEMIC CONTEXT:
{context}


Please provide a clear and accurate answer based
on the academic context above.
"""


    chat_completion = client.chat.completions.create(

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        model="openai/gpt-oss-120b",

        temperature=0.2
    )


    return chat_completion.choices[
        0
    ].message.content


# ============================================================
# USER QUESTION
# ============================================================

question = st.text_area(
    "💬 Ask your academic question",
    placeholder=(
        "Example: Explain the main concept discussed "
        "in the chapter about research methodology."
    ),
    height=120
)


ask_button = st.button(
    "🔍 Ask Assistant",
    type="primary",
    use_container_width=True
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a question first."
        )

        st.stop()


    with st.spinner(
        "Searching the academic knowledge base..."
    ):

        results = retrieve_documents(
            question,
            top_k
        )


    if not results:

        st.warning(
            "No relevant information was found."
        )

        st.stop()


    # Build context

    context = build_context(
        results
    )


    # Generate answer

    with st.spinner(
        "Generating academic answer..."
    ):

        answer = generate_answer(
            question,
            context
        )


    if answer:

        st.subheader("📖 Answer")

        st.markdown(answer)


        # ====================================================
        # SOURCE TRACEABILITY
        # ====================================================

        st.divider()

        st.subheader(
            "📚 Sources & Traceability"
        )


        for number, result in enumerate(
            results,
            start=1
        ):

            file_name = result.get(
                "file_name",
                result.get(
                    "source",
                    "Unknown"
                )
            )

            page_number = result.get(
                "page_number",
                "Unknown"
            )

            chunk_number = result.get(
                "chunk_number",
                "Unknown"
            )

            similarity = result.get(
                "similarity_score",
                0
            )


            with st.expander(
                f"Source {number}: "
                f"{file_name} — Page {page_number}"
            ):

                st.write(
                    f"**File:** {file_name}"
                )

                st.write(
                    f"**Page:** {page_number}"
                )

                st.write(
                    f"**Chunk:** {chunk_number}"
                )

                st.write(
                    f"**Similarity:** "
                    f"{similarity:.4f}"
                )

                st.write(
                    "**Retrieved text:**"
                )

                st.info(
                    result.get(
                        "text",
                        ""
                    )
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "University Academic Knowledge Assistant • "
    "RAG + FAISS + Sentence Transformers + Groq"
)
