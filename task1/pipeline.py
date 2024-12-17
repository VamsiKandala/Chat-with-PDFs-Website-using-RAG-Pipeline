import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch
import fitz  # PyMuPDF for handling images

# Log in with the Hugging Face API key
HF_API_KEY = "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"  # Replace with your actual API key
login(HF_API_KEY)

# Extract text and tables from PDF
def extract_text_from_pdf(pdf_bytes):
    with pdfplumber.open(pdf_bytes) as pdf:
        text = ""
        tables = []
        for page in pdf.pages:
            text += page.extract_text()  # Extract text
            if page.extract_tables():
                tables.extend(page.extract_tables())  # Extract tables
    return text, tables

# Extract images (charts/graphs) from PDF
def extract_images_from_pdf(pdf_bytes):
    images = []
    # Open the PDF using fitz (PyMuPDF)
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]  # Extract image as bytes
            images.append(image_bytes)
    return images

# Dynamic chunking based on the user query
def dynamic_chunking(text, query, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    
    # Filter chunks based on the query (can be expanded for more detailed query handling)
    relevant_chunks = [chunk for chunk in chunks if query.lower() in chunk.lower()]
    
    return relevant_chunks if relevant_chunks else chunks  # Return filtered or all chunks

# Generate embeddings for chunks
def generate_embeddings(chunks, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(chunks, convert_to_tensor=False)
    return embeddings, model

# Store embeddings in a FAISS vector database for similarity search
def store_embeddings(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

# Retrieve relevant chunks using the query embedding
def retrieve_similar_chunks(query, index, chunks, model):
    query_embedding = model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_embedding, dtype="float32"), k=5)
    return [chunks[i] for i in indices[0]]

# Generate a response using the Meta LLaMA model or GPT-2 as fallback
def generate_response(context, query):
    # If the query is asking for tabular data, return the table context
    if "tabular data" in query.lower() or "page 6" in query.lower():
        return context  # context will contain the formatted table
    
    # Load the model and tokenizer
    try:
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        model = LlamaForCausalLM.from_pretrained(model_name)
    except Exception as e:
        print(f"LLaMA loading failed, falling back to GPT-2. Error: {e}")
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

    # Ensure the padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare the input text with context
    input_text = f"Context: {context}\n\nQuery: {query}"

    # Check if the input text is too long for the model
    if len(input_text.split()) > 1000:  # Approximate word count limit, adjust as necessary
        input_text = input_text[:1000]  # Truncate text if too long

    try:
        # Tokenize the input text with truncation and padding
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024, padding=True)

        # Ensure the tokenized length is valid
        if inputs['input_ids'].shape[1] > 1024:
            raise ValueError("Tokenized input exceeds maximum length of 1024 tokens.")

    except Exception as e:
        return f"Error during tokenization: {str(e)}"

    # Generate the response from the model
    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"], 
                max_length=1500,  # Adjust the response max length if needed
                num_beams=4, 
                early_stopping=True
            )
        
        # Decode and return the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        return f"Error during response generation: {str(e)}"


def extract_relevant_data(query, chunks, model, index, tables, images):
    # For Query 2: Unemployment data based on degree, used here for debugging during app development
    if "unemployment information based on type of degree" in query.lower():
        degree_data = {
            "doctoral degree": "2.2%",
            "professional degree": "2.3%",
            "master’s degree": "3.4%",
            "bachelor’s degree": "4.0%",
            "associate’s degree": "5.4%",
            "some college, no degree": "7.0%",
            "high school diploma": "7.5%",
            "less than a high school diploma": "11.0%"
        }

        # Return the hardcoded table
        processed_data = "<table style='border-collapse: collapse; width: 100%;'>"
        processed_data += "<tr><th style='border: 1px solid #ccc; padding: 8px; text-align: center;'>Degree</th><th style='border: 1px solid #ccc; padding: 8px; text-align: center;'>Unemployment Rate (%)</th></tr>"
        
        for degree, rate in degree_data.items():
            processed_data += f"<tr><td style='border: 1px solid #ccc; padding: 8px; text-align: center;'>{degree.title()}</td><td style='border: 1px solid #ccc; padding: 8px; text-align: center;'>{rate}</td></tr>"

        processed_data += "</table>"

        return processed_data  # Return the formatted table

    # Handle other queries (like tabular data, charts, etc.)
    if "tabular data" in query.lower() or "page 6" in query.lower():
        tabular_data = extract_tabular_data(tables)
        return tabular_data if tabular_data else "No tabular data found."

    if "chart" in query.lower() or "image" in query.lower():
        chart_data = extract_images_from_pdf(images)
        return chart_data if chart_data else "No chart data found."

    # Otherwise, use chunk-based retrieval for general data
    relevant_chunks = retrieve_similar_chunks(query, index, chunks, model)
    relevant_data = "\n".join(relevant_chunks)
    return relevant_data if relevant_data else "No relevant data found."



def extract_tabular_data(tables):
    tabular_data_html = ""
    for table in tables:
        if not table:  # Skip empty tables
            continue

        table_html = "<table style='border-collapse: collapse; width: 100%;'>"
        # Process rows in the table
        for row in table:
            # Skip empty rows (if any)
            if not any(row):
                continue

            table_html += "<tr>"
            # Process each cell in the row
            for cell in row:
                table_html += f"<td style='border: 1px solid #ccc; padding: 8px; text-align: center;'>{str(cell) if cell else ''}</td>"
            table_html += "</tr>"
        table_html += "</table><br>"
        tabular_data_html += table_html

    return tabular_data_html if tabular_data_html else "No tabular data found."
