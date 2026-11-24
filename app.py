
import os
import re
import io
import time
import hashlib

import numpy as np
import streamlit as st

from dotenv import load_dotenv
from google import genai
from google.genai import types

from file_extractor import extract_file_content

from database import (
    initialize_database,
    save_document_chunks,
    get_document_chunks,
    get_document_filename,
    get_all_documents
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ENV_PATH = os.path.join(BASE_DIR, ".env")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")

os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

load_dotenv(ENV_PATH, override=True)

initialize_database()

EMBEDDING_MODEL = "gemini-embedding-001"
ANSWER_MODEL = "gemini-3.5-flash-lite"

EMBEDDING_DIMENSION = 768

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

SUPPORTED_EXTENSIONS = [
    "pdf",
    "docx",
    "csv",
    "xlsx",
    "pptx",
    "txt",
    "md",
    "json",
    "jpg",
    "jpeg",
    "png",
    "webp",
    "bmp",
    "tiff",
    "tif"
]


# ==========================================
# 2. STREAMLIT PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Intelligent Document QA",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 750;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #888888;
        font-size: 1rem;
        margin-bottom: 30px;
    }

    .ready-box {
        padding: 16px;
        border-radius: 12px;
        background-color: rgba(40, 167, 69, 0.10);
        border: 1px solid rgba(40, 167, 69, 0.35);
        margin-top: 15px;
        margin-bottom: 20px;
    }

    .chat-heading {
        font-size: 1.5rem;
        font-weight: 650;
        margin-top: 25px;
        margin-bottom: 5px;
    }

    .muted-text {
        color: #888888;
        font-size: 0.95rem;
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 44px;
        font-weight: 600;
    }

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. APPLICATION HEADER
# ==========================================

st.markdown(
    '<div class="main-title">📚 Intelligent Document Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload your documents and get instant answers with AI.'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================
# 4. SESSION STATE
# ==========================================

SESSION_DEFAULTS = {
    "active_file_hash": None,
    "document_id": None,
    "vector_path": None,
    "embeddings": None,
    "answer": None,
    "last_question": None,
    "search_results": [],
    "history_selection": None,
    "processing_error": None
}

for key, value in SESSION_DEFAULTS.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ==========================================
# 5. GEMINI CLIENT
# ==========================================

def get_gemini_client():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found. "
            "Please check your .env file."
        )

    return genai.Client(api_key=api_key)


# ==========================================
# 6. TEXT PREPROCESSING
# ==========================================

def preprocess_text(text):

    if not text:
        return ""

    text = text.replace("\x00", " ")

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ==========================================
# 7. TEXT CHUNKING
# ==========================================

def create_chunks(sections):

    chunks = []

    chunk_number = 1

    for section in sections:

        text = preprocess_text(section.get("text", ""))

        location = section.get(
            "location",
            "Document"
        )

        if not text:
            continue

        words = text.split()

        if not words:
            continue

        start = 0

        while start < len(words):

            end = start + CHUNK_SIZE

            chunk_words = words[start:end]

            chunk_text = " ".join(chunk_words)

            # Preserve the original source location.
            page_number = 0

            page_match = re.match(
                r"Page\s+(\d+)",
                str(location),
                re.IGNORECASE
            )

            if page_match:
                page_number = int(page_match.group(1))

            chunks.append({
                "chunk_number": chunk_number,
                "page_number": page_number,
                "source_location": str(location),
                "text": chunk_text
            })

            chunk_number += 1

            if end >= len(words):
                break

            start = end - CHUNK_OVERLAP

    return chunks


# ==========================================
# 8. GENERATE DOCUMENT EMBEDDING
# ==========================================

def generate_embedding(client, text):

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=EMBEDDING_DIMENSION
        )
    )

    return response.embeddings[0].values


# ==========================================
# 9. GENERATE ALL EMBEDDINGS
# ==========================================



def generate_all_embeddings(client, chunks, progress_bar, status):

    embeddings = []

    total_chunks = len(chunks)

    # Smaller batches to reduce token usage per request
    batch_size = 1

    # Maximum number of retries for quota errors
    max_retries = 4

    total_batches = (
        total_chunks + batch_size - 1
    ) // batch_size

    for batch_number, start in enumerate(
        range(0, total_chunks, batch_size),
        start=1
    ):

        end = min(start + batch_size, total_chunks)

        batch_texts = [
            chunk["text"]
            for chunk in chunks[start:end]
        ]

        status.write(
            f"Generating embeddings: "
            f"Batch {batch_number}/{total_batches} "
            f"({end}/{total_chunks} chunks)"
        )

        for attempt in range(max_retries + 1):

            try:

                response = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch_texts,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=EMBEDDING_DIMENSION
                    )
                )

                batch_embeddings = [
                    embedding.values
                    for embedding in response.embeddings
                ]

                if len(batch_embeddings) != len(batch_texts):
                    raise ValueError(
                        "Gemini returned an unexpected number of embeddings."
                    )

                embeddings.extend(batch_embeddings)

                break

            except Exception as error:

                error_message = str(error)

                is_quota_error = (
                    "429" in error_message
                    or "RESOURCE_EXHAUSTED" in error_message
                )

                if not is_quota_error:
                    raise

                if attempt == max_retries:
                    raise RuntimeError(
                        "Gemini embedding quota is still exhausted. "
                        "Please wait before uploading again."
                    ) from error

                # Wait progressively longer before retrying
                wait_seconds = 65

                status.warning(
                    f"Gemini rate limit reached. "
                    f"Retrying batch {batch_number} in "
                    f"{wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)

        progress_bar.progress(
            end / total_chunks
        )

        # Space out requests to control token usage per minute
        if end < total_chunks:
            time.sleep(3)

    return np.array(
        embeddings,
        dtype=np.float32
    )




# ==========================================
# 10. SAVE DOCUMENT AND EMBEDDINGS
# ==========================================

def save_embeddings(
    embeddings,
    filename,
    file_hash,
    chunks
):

    safe_filename = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        os.path.splitext(filename)[0]
    )

    document_id = (
        f"{safe_filename}_{file_hash[:12]}"
    )

    vector_path = os.path.join(
        VECTOR_STORE_DIR,
        f"{document_id}_vectors.npy"
    )

    np.save(
        vector_path,
        embeddings
    )

    save_document_chunks(
        document_id,
        filename,
        chunks
    )

    return document_id, vector_path


# ==========================================
# 11. COSINE SIMILARITY
# ==========================================

def cosine_similarity(vector_a, vector_b):

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(
        np.dot(vector_a, vector_b) /
        (norm_a * norm_b)
    )


# ==========================================
# 12. SEMANTIC SEARCH
# ==========================================

def semantic_search(
    question,
    client,
    document_id,
    top_k=3
):

    vector_path = os.path.join(
        VECTOR_STORE_DIR,
        f"{document_id}_vectors.npy"
    )

    if not os.path.isfile(vector_path):
        raise FileNotFoundError(
            "The embedding file for this document is missing."
        )

    document_vectors = np.load(
        vector_path,
        allow_pickle=False
    )

    chunks = get_document_chunks(document_id)

    if not chunks:
        raise ValueError(
            "No document text was found in the database."
        )

    if len(document_vectors) != len(chunks):
        raise ValueError(
            "The saved embeddings and document chunks do not match."
        )

    if document_vectors.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(
            "The embedding dimensions are incompatible."
        )

    query_response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBEDDING_DIMENSION
        )
    )

    question_vector = np.array(
        query_response.embeddings[0].values,
        dtype=np.float32
    )

    similarities = []

    for index, document_vector in enumerate(document_vectors):

        score = cosine_similarity(
            question_vector,
            document_vector
        )

        similarities.append({
            "chunk": chunks[index],
            "similarity": score
        })

    similarities.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    filename = get_document_filename(document_id)

    results = []

    for item in similarities[:top_k]:

        chunk = item["chunk"]

        results.append({
            "document_id": document_id,
            "file_name": filename,
            "chunk_number": chunk["chunk_number"],
            "page_number": chunk["page_number"],
            "source_location": chunk.get(
                "source_location",
                "Document"
            ),
            "text": chunk["text"],
            "similarity": item["similarity"]
        })

    return results


# ==========================================
# 13. GENERATE ANSWER
# ==========================================

def generate_answer(question, search_results, client):

    context_parts = []

    for result in search_results:

        context_parts.append(
            f"""
Document: {result['file_name']}
Location: {result['source_location']}
Content:
{result['text']}
"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an intelligent document question-answering assistant.

Answer the user's question using ONLY the provided document context.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not invent facts or information.
3. If the answer is not available in the context, clearly say:
   "The uploaded document does not contain enough information
   to answer this question."
4. Give a clear, accurate, and well-structured answer.
5. Use simple language.
6. When useful, include the source location in this format:
   [Source: filename, location]

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    response = client.models.generate_content(
        model=ANSWER_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=512
        )
    )

    if not response.text:
        raise ValueError(
            "The AI model returned an empty answer."
        )

    return response.text.strip()


# ==========================================
# 14. LOAD PREVIOUS DOCUMENT
# ==========================================

def load_previous_document(document_id):

    vector_path = os.path.join(
        VECTOR_STORE_DIR,
        f"{document_id}_vectors.npy"
    )

    if not os.path.isfile(vector_path):
        raise FileNotFoundError(
            "The embedding file for this document is missing. "
            "Please upload the document again."
        )

    vectors = np.load(
        vector_path,
        allow_pickle=False
    )

    chunks = get_document_chunks(document_id)

    if not chunks:
        raise ValueError(
            "No saved document chunks were found."
        )

    if len(vectors) != len(chunks):
        raise ValueError(
            "Saved document data is incomplete or incompatible."
        )

    if vectors.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(
            "Saved embeddings have an incompatible dimension."
        )

    st.session_state.document_id = document_id
    st.session_state.vector_path = vector_path
    st.session_state.embeddings = vectors
    st.session_state.active_file_hash = None

    st.session_state.answer = None
    st.session_state.last_question = None
    st.session_state.search_results = []



# ==========================================
# 15. DOCUMENT HISTORY
# ==========================================

st.sidebar.title("📂 Your Documents")

documents = get_all_documents()

document_options = {
    document["document_id"]: (
        f"{document['filename']} "
        f"({document['upload_date']})"
    )
    for document in documents
}

history_options = [
    "Upload a new document"
] + list(document_options.keys())


def format_history_option(option):

    if option == "Upload a new document":
        return option

    return document_options.get(
        option,
        "Unknown document"
    )


selected_history = st.sidebar.selectbox(
    "Document History",
    options=history_options,
    format_func=format_history_option,
    key="document_history_selector"
)


# Load a previous document only when selected.
if selected_history != "Upload a new document":

    if selected_history != st.session_state.document_id:

        try:

            load_previous_document(selected_history)

            st.sidebar.success(
                "Previous document loaded!"
            )

        except Exception as error:

            st.sidebar.error(str(error))


# IMPORTANT:
# Do not reset active_file_hash when
# "Upload a new document" is selected.
#
# Streamlit reruns the script after processing.
# Keeping the hash prevents repeated embedding generation.



# ==========================================
# 16. DOCUMENT UPLOAD
# ==========================================

st.subheader("📄 Upload Your Document")

st.write(
    "Upload a document to begin. "
    "We'll prepare it automatically so you can start chatting."
)

uploaded_file = st.file_uploader(
    "Choose a document",
    type=SUPPORTED_EXTENSIONS,
    accept_multiple_files=False,
    help="Supported formats include PDF, DOCX, CSV, XLSX, PPTX, TXT, MD, JSON and images."
)


# ==========================================
# 17. AUTOMATIC DOCUMENT PROCESSING
# ==========================================

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    file_hash = hashlib.sha256(
        file_bytes
    ).hexdigest()

    if file_hash != st.session_state.active_file_hash:

        st.session_state.answer = None
        st.session_state.last_question = None
        st.session_state.search_results = []

        st.session_state.document_id = None
        st.session_state.vector_path = None
        st.session_state.embeddings = None

        try:

            with st.status(
                "Preparing your document...",
                expanded=True
            ) as status:

                st.write("Reading your uploaded document...")

                # Extract content using file_extractor.py
                sections = extract_file_content(
                    uploaded_file
                )

                if not sections:
                    raise ValueError(
                        "No readable content was found in this document."
                    )

                st.write("Cleaning and processing the text...")

                chunks = create_chunks(sections)

                if not chunks:
                    raise ValueError(
                        "No usable text could be extracted from this document."
                    )

                st.write(
                    f"Extracted {len(chunks)} text chunks."
                )

                client = get_gemini_client()

                st.write(
                    "Generating AI embeddings automatically..."
                )

                progress_bar = st.progress(0)

                embedding_status = st.empty()

                embeddings = generate_all_embeddings(
                    client,
                    chunks,
                    progress_bar,
                    embedding_status
                )

                st.write(
                    "Saving your document..."
                )

                document_id, vector_path = save_embeddings(
                    embeddings,
                    uploaded_file.name,
                    file_hash,
                    chunks
                )

                st.session_state.active_file_hash = file_hash
                st.session_state.document_id = document_id
                st.session_state.vector_path = vector_path
                st.session_state.embeddings = embeddings

                status.update(
                    label="Your document is ready!",
                    state="complete",
                    expanded=False
                )

            st.success(
                "🎉 Your document has been processed successfully. "
                "You can now ask questions!"
            )

            st.rerun()

        except Exception as error:

            st.error(
                f"Document processing failed: {error}"
            )

            st.info(
                "Please check the document format and your Gemini API configuration, "
                "then try uploading again."
            )


# ==========================================
# 18. DOCUMENT READY STATUS
# ==========================================

if st.session_state.document_id:

    current_filename = get_document_filename(
        st.session_state.document_id
    )

    st.markdown(
        f"""
        <div class="ready-box">
            <strong>✅ Document ready for conversation</strong><br>
            <span>{current_filename}</span>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# 19. CHAT WITH YOUR DOCUMENT
# ==========================================

if st.session_state.document_id:

    st.markdown(
        '<div class="chat-heading">💬 Chat with Your Document</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="muted-text">'
        'Ask anything about your uploaded document. '
        'The AI will find relevant information and answer your question.'
        '</div>',
        unsafe_allow_html=True
    )

    with st.form("question_form"):

        question = st.text_input(
            "Your question",
            placeholder="e.g. What are the main objectives of this document?",
            label_visibility="collapsed"
        )

        col1, col2 = st.columns([1, 4])

        with col1:

            top_k = st.select_slider(
                "Answer context",
                options=[1, 2, 3, 4, 5],
                value=3
            )

        with col2:

            submit_question = st.form_submit_button(
                "✨ Get Answer",
                type="primary",
                use_container_width=True
            )

    if submit_question:

        if not question.strip():

            st.warning(
                "Please enter a question first."
            )

        else:

            try:

                client = get_gemini_client()

                with st.spinner(
                    "Finding information and preparing your answer..."
                ):

                    search_results = semantic_search(
                        question,
                        client,
                        st.session_state.document_id,
                        top_k=top_k
                    )

                    answer = generate_answer(
                        question,
                        search_results,
                        client
                    )

                    st.session_state.last_question = question
                    st.session_state.answer = answer
                    st.session_state.search_results = search_results

            except Exception as error:

                st.error(
                    f"Unable to generate an answer: {error}"
                )

    # Display the answer only.
    # Retrieved sources and technical search information stay hidden.

    if st.session_state.answer:

        st.divider()

        st.markdown("### 🤖 Answer")

        st.markdown(
            st.session_state.answer
        )

        st.caption(
            f"Question: {st.session_state.last_question}"
        )

else:

    st.info(
        "👆 Upload a document above or select a previous document "
        "from Document History to start chatting."
    )


# ==========================================
# 20. FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:gray; font-size:0.85rem;">
        Intelligent Document Assistant · Powered by Gemini AI
    </div>
    """,
    unsafe_allow_html=True
)
