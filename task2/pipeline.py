import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch

# Log in with the Hugging Face API key
HF_API_KEY = "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"  # Replace with your actual API key
login(HF_API_KEY)

# Scrape website content
def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Extract text
    paragraphs = soup.find_all("p")
    text = " ".join([para.get_text() for para in paragraphs])
    
    # Extract tables
    tables = []
    for table in soup.find_all("table"):
        rows = []
        for row in table.find_all("tr"):
            cells = [cell.get_text(strip=True) for cell in row.find_all(["td", "th"])]
            rows.append(cells)
        tables.append(rows)
    
    # Extract images
    images = [img["src"] for img in soup.find_all("img", src=True)]
    
    return text, tables, images

# Chunking the data
def dynamic_chunking(text, chunk_size=500, chunk_overlap=50):
    soup = BeautifulSoup(text, "html.parser")
    sections = []
    
    for header in soup.find_all(["h1", "h2", "h3"]):
        section_text = []
        for sibling in header.find_next_siblings():
            if sibling.name in ["h1", "h2", "h3"]:
                break
            section_text.append(sibling.get_text())
        sections.append(f"{header.get_text()}\n{''.join(section_text)}")
    
    if not sections:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_text(text)

    return sections

# Generate embeddings
def generate_embeddings(chunks, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(chunks, convert_to_tensor=False)
    return embeddings, model

# Store embeddings in FAISS
def store_embeddings(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

# Retrieve relevant chunks
def retrieve_similar_chunks(query, index, chunks, model, top_k=5):
    query_embedding = model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_embedding, dtype="float32"), k=top_k)

    retrieved_chunks = [chunks[i] for i in indices[0] if query.lower() in chunks[i].lower()]
    return retrieved_chunks if retrieved_chunks else [chunks[i] for i in indices[0]]

# Generate response
def generate_response(context, query):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    input_text = f"Context: {context}\n\nQuery: {query}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024, padding=True)
    outputs = model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract relevant data
def extract_relevant_data(query, chunks, model, index, tables, images):
    if not chunks or not index:
        return "No data available. Please process a website URL first."

    relevant_chunks = retrieve_similar_chunks(query, index, chunks, model)
    relevant_data = "\n".join(relevant_chunks[:1])  # Select the most relevant chunk

    # Post-processing to clean up text
    relevant_data = " ".join(relevant_data.split())  # Remove extra spaces/newlines
    if len(relevant_data) > 500:
        relevant_data = relevant_data[:500] + "..."

    if "table" in query.lower():
        return extract_tabular_data(tables)
    elif "image" in query.lower():
        return "\n".join(images) if images else "No images found."

    return relevant_data if relevant_data else "No relevant data found."

# Extract tabular data
def extract_tabular_data(tables):
    html = ""
    for table in tables:
        table_html = "<table border='1'>"
        for row in table:
            table_html += "<tr>" + "".join([f"<td>{cell}</td>" for cell in row]) + "</tr>"
        table_html += "</table><br>"
        html += table_html
    return html if html else "No tables found."
