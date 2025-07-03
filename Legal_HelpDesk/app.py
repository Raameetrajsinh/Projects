import streamlit as st
import pdfplumber
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from deep_translator import GoogleTranslator

st.set_page_config(page_title="Legal Assistant", layout="wide")


# Load TinyLlama
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model.eval()
    return tokenizer, model

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Chunk into smaller pieces
def chunk_text(text, max_chunk_size=500):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Embedding and FAISS indexing
@st.cache_resource
def embed_and_index(chunks):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embedder, embeddings

# Search top-k relevant chunks
def search(query, embedder, index, chunks, top_k=5):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, top_k)
    return [chunks[i] for i in I[0]]


def translate_text(text, target_language):
    return GoogleTranslator(source="auto", target=target_language).translate(text)

# Run the model on context
def run_llm(tokenizer, model, context, question):
    prompt = f"""
You are a legal assistant. Based on the Constitution of Iran, answer the question using only information from the context below. Do not guess or add unrelated sections.

Context:    
{context}

Question:
{question}

Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Answer:")[-1].strip()




st.title("📘 AI Powered Legal HelpDesk For Iran")

with st.sidebar:
    st.header("📤 Upload PDF")
    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])



if uploaded_pdf:
    with st.spinner("Extracting and indexing..."):
        text = extract_text_from_pdf(uploaded_pdf)
        chunks = chunk_text(text)
        index, embedder, _ = embed_and_index(chunks)
        tokenizer, model = load_llm()
        st.success("✅ Document processed! Ask your question below.")

    # Language selection 
    input_lang = st.radio("Input Language", ["English", "Persian"], index=0)
    output_lang = st.radio("Output Language", ["English", "Persian"], index=0)

    
    if input_lang == "Persian":
        question = st.text_input("سوال خود را بنویسید:", key="query_input")
        st.markdown("""
            <style>
            .stTextInput > div > div > input {
                direction: rtl;
                text-align: right;
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        question = st.text_input("💬 Ask a question:", key="query_input")

    # Answer generation
    if question.strip():
        with st.spinner("Generating answer..."):
        
            translated_question = translate_text(question, "en") if input_lang == "Persian" else question

    
            relevant_chunks = search(translated_question, embedder, index, chunks)
            context = "\n".join(relevant_chunks)
            answer = run_llm(tokenizer, model, context, translated_question)

            
            if output_lang == "Persian":
                answer = translate_text(answer, "fa")
                st.markdown("### 🧠 پاسخ")
                st.markdown(f"<div dir='rtl' style='text-align: right;'>{answer}</div>", unsafe_allow_html=True)
            else:
                st.markdown("### 🧠 Answer")
                st.write(answer)

            # Context visibility
            # with st.expander("📄 Show Retrieved Context"):
            #     st.text(context)