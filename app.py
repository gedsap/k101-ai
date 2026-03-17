import os
import json
import math
import heapq
import urllib.request
st.title("...")import streamlit as st

def safe_load_index():
    try:
        ensure_index_file()
        return True
    except Exception as e:
        st.error(f"Index failed to load: {e}")
        return False

from openai import OpenAI

# -----------------------
# CONFIG
# -----------------------

INDEX_URL = "https://huggingface.co/datasets/gesap/K101.Ai/resolve/main/lecture_index.json"
INDEX_FILE = "lecture_index.json"

TOP_K = 25
MAX_HISTORY = 10
COMPRESS_MODEL = "gpt-4.1-mini"
ANSWER_MODEL = "gpt-4.1"

# -----------------------
# PAGE SETUP
# -----------------------

st.set_page_config(page_title="Lecture AI", layout="wide")

# -----------------------
# PASSWORD PROTECTION
# -----------------------

def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if st.session_state.password_correct:
        return True

    st.title("Kohelet AI")

    if "index_loaded" not in st.session_state:
    with st.spinner("Loading lecture index..."):
        success = safe_load_index()
        st.session_state["index_loaded"] = success

    st.write("The Thought of Creation!")

    password = st.text_input("Password", type="password")

    if password:
        if password == st.secrets["APP_PASSWORD"]:
            st.session_state.password_correct = True
            st.rerun()
        else:
            st.error("Incorrect password.")
            st.stop()

    st.stop()

check_password()

# -----------------------
# OPENAI CLIENT
# -----------------------

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -----------------------
# DOWNLOAD INDEX IF NEEDED
# -----------------------

def # ensure_index_file():
    if os.path.exists(INDEX_FILE):
        return

    st.title("Lecture AI")
    st.info("Preparing lecture index. First launch may take a minute or two.")

    with st.spinner("Downloading lecture index..."):
        urllib.request.urlretrieve(INDEX_URL, INDEX_FILE)

ensure_index_file()

# -----------------------
# LOAD INDEX
# -----------------------

@st.cache_data(show_spinner="Loading lecture index...")
def load_index():
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

index = load_index()

# -----------------------
# MATH
# -----------------------

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

# -----------------------
# RETRIEVAL
# -----------------------

def retrieve_top_chunks(question, top_k=TOP_K):
    qemb = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    scored = []
    for item in index:
        sim = cosine(qemb, item["embedding"])
        scored.append((sim, item["text"]))

    top_chunks = [x[1] for x in heapq.nlargest(top_k, scored, key=lambda 
x: x[0])]
    return top_chunks

def compress_context(question, chunks):
    compression_prompt = f"""
Select the passages from these lecture excerpts that are most useful for 
answering the user's question.

User question:
{question}

Lecture excerpts:
{chunks}
"""
    response = client.responses.create(
        model=COMPRESS_MODEL,
        input=compression_prompt
    )
    return response.output[0].content[0].text

# -----------------------
# CHAT STATE
# -----------------------

if "history" not in st.session_state:
    st.session_state.history = []

st.session_state.history = st.session_state.history[-MAX_HISTORY:]

# -----------------------
# UI
# -----------------------

st.title("Lecture AI")

col1, col2 = st.columns([1, 5])
with col1:
    if st.button("Reset conversation"):
        st.session_state.history = []
        st.rerun()

with col2:
    st.caption("Private lecture search and Q&A")

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -----------------------
# CHAT INPUT
# -----------------------

question = st.chat_input("Ask something about the lectures")

if question:
    st.session_state.history.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()

        try:
            with st.spinner("Searching lectures..."):
                top_chunks = retrieve_top_chunks(question)
                compressed = compress_context(question, top_chunks)

            full_text = ""

            with client.responses.stream(
                model=ANSWER_MODEL,
                input=f"""
Answer the user's question using the lecture material below.

Lecture material:
{compressed}

Question:
{question}
"""
            ) as stream:

                for event in stream:
                    if event.type == "response.output_text.delta":
                        full_text += event.delta
                        placeholder.write(full_text)

            if not full_text.strip():
                full_text = "No answer was generated."
                placeholder.write(full_text)

        except Exception as e:
            full_text = f"Error: {e}"
            placeholder.error(full_text)

    st.session_state.history.append({"role": "assistant", "content": full_text})
