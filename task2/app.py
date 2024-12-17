import streamlit as st
from pipeline import (
    scrape_website,
    dynamic_chunking,
    generate_embeddings,
    store_embeddings,
    retrieve_similar_chunks,
    extract_relevant_data,
)
from huggingface_hub import login
import numpy as np

# Log in with Hugging Face API key
HF_API_KEY = "[Your_API_Key]"  # Replace with your actual API key
login(HF_API_KEY)

# Streamlit App Configuration
st.set_page_config(page_title="Chat with Websites using RAG Pipeline", layout="wide")
st.title("Chat with Websites using RAG Pipeline")

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

if 'query_input' not in st.session_state:
    st.session_state.query_input = ""

if 'chunks' not in st.session_state:
    st.session_state.chunks = None

if 'model' not in st.session_state:
    st.session_state.model = None

if 'index' not in st.session_state:
    st.session_state.index = None

if 'tables' not in st.session_state:
    st.session_state.tables = None

if 'images' not in st.session_state:
    st.session_state.images = None

# Sidebar for website URL input
url_input = st.sidebar.text_input("Enter Website URL", placeholder="https://example.com")
process_url = st.sidebar.button("Process URL")

# Process the provided website URL
if process_url and url_input:
    with st.spinner("Scraping the website..."):
        text, tables, images = scrape_website(url_input)
        st.session_state.tables = tables
        st.session_state.images = images
        st.sidebar.success("Website data scraped successfully!")

    with st.spinner("Generating embeddings..."):
        chunks = dynamic_chunking(text)
        st.session_state.chunks = chunks

        embeddings, model = generate_embeddings(chunks)
        st.session_state.model = model

        index = store_embeddings(np.array(embeddings, dtype="float32"))
        st.session_state.index = index
        st.sidebar.success("Embeddings generated and stored!")

# Main UI layout (Chat History and Input)
st.markdown(
    """
    <style>
        .input-container { width: 100%; max-width: 600px; margin: 20px auto; text-align: center; }
        .history-container { display: flex; flex-direction: column; width: 100%; max-width: 600px; margin: 20px auto; padding: 10px; background-color: #f0f0f0; border-radius: 8px; overflow-y: auto; max-height: 400px; }
        .message { margin-bottom: 15px; padding: 10px; background-color: #ffffff; border-radius: 5px; border: 1px solid #ccc; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Chat History
with st.container():
    st.markdown("<div class='history-container'>", unsafe_allow_html=True)
    for chat in st.session_state.history:
        st.markdown(
            f"<div class='message'><strong>{chat['role'].capitalize()}:</strong> {chat['message']}</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

# Input Query
with st.container():
    st.markdown("<div class='input-container'>", unsafe_allow_html=True)
    query = st.text_input(
        "Enter your query:", value=st.session_state.query_input, key="unique_query_input"
    )
    run_query = st.button("Run Query")
    st.markdown("</div>", unsafe_allow_html=True)

# Handle User Query
if run_query and query:
    if st.session_state.chunks and st.session_state.index:
        with st.spinner("Processing your query..."):
            relevant_data = extract_relevant_data(
                query, 
                st.session_state.chunks, 
                st.session_state.model, 
                st.session_state.index, 
                st.session_state.tables, 
                st.session_state.images
            )
            st.session_state.history.append({"role": "user", "message": query})
            st.session_state.history.append({"role": "bot", "message": relevant_data})

        # Display Response
        if '<table' in relevant_data:
            st.markdown(relevant_data, unsafe_allow_html=True)
        else:
            st.markdown(f"**Bot:** {relevant_data}")
    else:
        st.error("Please process a website URL before asking queries.")
