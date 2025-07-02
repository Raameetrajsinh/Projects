import os
import streamlit as st
import pandas as pd
import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


st.set_page_config(page_title="SQL Assistant")

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load slim-sql model
@st.cache_resource
def load_sql_model():
    tokenizer = AutoTokenizer.from_pretrained("llmware/slim-sql-1b-v0")
    model = AutoModelForCausalLM.from_pretrained("llmware/slim-sql-1b-v0")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        temperature=0.3,
        pad_token_id=tokenizer.eos_token_id
    )
    return pipe

llm = load_sql_model()

# Database path
@st.cache_data
def get_db_path():
    path = "students.db"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Database file '{path}' not found.")
    return path

db_path = get_db_path()

# Load schema
@st.cache_data(ttl=86400)
def load_schema_descriptions():
    with sqlite3.connect(db_path) as conn:
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
        descriptions = []
        for table in tables['name']:
            cols = pd.read_sql_query(f"PRAGMA table_info({table});", conn)
            col_names = ", ".join(cols['name'])
            desc = f"Table: {table}. Columns: {col_names}."
            descriptions.append(desc)
        return descriptions

schema_descriptions = load_schema_descriptions()

# FAISS index
@st.cache_data(ttl=86400) 
def build_faiss_index(descriptions):
    embeddings = embedder.encode(descriptions, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

index, embeddings = build_faiss_index(schema_descriptions)

st.title("SQL Query Generator")
st.markdown("Ask about your Student database!")

user_query = st.text_input("Ask a question:", placeholder="e.g. List all students")

if user_query:
    with st.spinner("🔍 Retrieving relevant schema..."):
        query_emb = embedder.encode([user_query], convert_to_numpy=True)
        D, I = index.search(query_emb, k=3)
        context = "\n".join([schema_descriptions[i] for i in I[0]])

    
    prompt = f"<human>: Context:\n{context}\nQuestion: {user_query}\n<bot>:"

    with st.spinner("🤖 Generating SQL query..."):
        output = llm(prompt)
        sql_query = output[0]["generated_text"].split("<bot>:")[-1].strip()
        st.success("✅ SQL Query Generated")
        st.code(sql_query, language="sql")

    
    if st.button("▶️ Execute Query"):
        if not sql_query.lower().startswith("select"):
            st.error("❌ Only SELECT queries are allowed for safety.")
        else:
            try:
                with sqlite3.connect(db_path, check_same_thread=False) as conn:
                    result_df = pd.read_sql_query(sql_query, conn)
                st.success("✅ Query Executed")
                st.dataframe(result_df)
            except Exception as e:
                st.error(f"❌ SQL Execution Error: {e}")