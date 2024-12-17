import streamlit as st
from pipeline import extract_text_from_pdf, dynamic_chunking, generate_embeddings, store_embeddings, retrieve_similar_chunks, generate_response, extract_relevant_data, extract_images_from_pdf
from huggingface_hub import login
import numpy as np
from io import BytesIO

# Log in with the Hugging Face API key
HF_API_KEY = "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"  # Replace with your actual API key
login(HF_API_KEY)

# Streamlit App Configuration
st.set_page_config(page_title="Chat with PDFs using RAG Pipeline", layout="wide")
st.title("Chat with PDFs using RAG Pipeline")

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

if 'query_input' not in st.session_state:
    st.session_state.query_input = ""

# Sidebar for PDF file uploader
uploaded_files = st.sidebar.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")

# Process uploaded files
if uploaded_files:
    st.sidebar.write(f"Uploaded {len(uploaded_files)} file(s).")
    pdf_texts = []
    chunks = []
    tables = []
    images = []

    for file in uploaded_files:
        # Process each uploaded PDF file directly from memory
        pdf_bytes = BytesIO(file.read())  # Convert file content to BytesIO
        text, table_data = extract_text_from_pdf(pdf_bytes)  # Pass BytesIO to extraction function
        pdf_texts.append(text)
        tables.extend(table_data)
        chunks.extend(dynamic_chunking(text, st.session_state.query_input))  # Chunk based on query
        images.extend(extract_images_from_pdf(pdf_bytes))  # Extract images (charts/graphs)

    # Display success messages in the sidebar
    st.sidebar.success("Text extracted and chunked successfully!")
    
    # Generate embeddings for chunks and store them
    embeddings, model = generate_embeddings(chunks)
    index = store_embeddings(np.array(embeddings, dtype="float32"))
    st.sidebar.success("Embeddings generated and stored!")

# Main UI layout (All responses appear above the input area)
st.markdown(
    """
    <style>
        .input-container {
            width: 100%;
            max-width: 600px;
            margin: 20px auto;
            text-align: center;
        }
        .history-container {
            display: flex;
            flex-direction: column;
            width: 100%;
            max-width: 600px;
            margin: 20px auto;
            padding: 10px;
            background-color: #f0f0f0;
            border-radius: 8px;
            overflow-y: auto;
            max-height: 400px;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px;
            background-color: #ffffff;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display chat history above the input area
with st.container():
    st.markdown("<div class='history-container'>", unsafe_allow_html=True)
    for chat in st.session_state.history:
        st.markdown(
            f"<div class='message'><strong>{chat['role'].capitalize()}:</strong> {chat['message']}</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

# Input area for query (below chat history)
with st.container():
    st.markdown("<div class='input-container'>", unsafe_allow_html=True)
    query = st.text_input(
        "Enter your query:", value=st.session_state.query_input, key="unique_query_input"
    )
    run_query = st.button("Run Query")
    st.markdown("</div>", unsafe_allow_html=True)

# Process user query
if run_query and query:
    with st.spinner("Processing your query..."):
        # Directly extract data based on the query
        relevant_data = extract_relevant_data(query, chunks, model, index, tables, images)

        # Append the user query and bot response to session state history
        st.session_state.history.append({"role": "user", "message": query})
        st.session_state.history.append({"role": "bot", "message": relevant_data})

        # Clear query input for next interaction
        st.session_state.query_input = ""  # This line will clear the input field

        # Display the response
        st.markdown(f"**User:** {query}")

        # Check if the response contains table data
        if '<table' in relevant_data:  # Check if the response contains a table
            st.markdown(relevant_data, unsafe_allow_html=True)  # Render HTML table
        else:
            st.markdown(f"**Bot:** {relevant_data}")  # Otherwise, display the response as text

        # Reset the query input field to be empty after submission
        st.session_state.query_input = ""  # This line will clear the input field
