import streamlit as st
import tempfile
import os
import pandas as pd
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from utils import *  

load_dotenv()

st.set_page_config(page_title="RFP Clarification Bot", layout="wide")
st.title("RFP Pre-Bid Clarification Bot")

# Load model
@st.cache_resource
def load_model():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

generator = load_model()

# Upload RFP
uploaded_file = st.file_uploader("Upload your RFP PDF", type=["pdf"])

if uploaded_file:
    file_bytes = uploaded_file.read()
    namespace = uploaded_file.name.replace(".pdf", "").replace(" ", "_").lower()
    cleaned_pdf_name = namespace

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    index = ensure_index_exists()

    existing_namespaces = index.describe_index_stats().namespaces.keys()
    if namespace in existing_namespaces:
        st.info(f"This RFP (`{namespace}`) has already been uploaded and indexed.")
    else:
        st.write("Processing PDF...")
        chunks = process_pdf_chunks(tmp_path)
        embeddings = embed_chunks(chunks)
        store_vectors_to_namespace(index, namespace, chunks, embeddings, cleaned_pdf_name)
        st.success(f"RFP uploaded and indexed successfully under namespace `{namespace}`!")


    excel_file = st.file_uploader("Upload Excel file", type=["xlsx"])

    if excel_file:
        try:
            df = pd.read_excel(excel_file)
            st.write("Uploaded Excel:")
            st.dataframe(df)

            if st.button("Run Queries"):
                with st.spinner("Processing queries..."):
                    results = []

                    for question in df["Query Raised"]:
                        try:
                            matches = query_index(index, namespace, question)
                            context = "\n".join(matches[:3]) if matches else ""

                            if context.strip():
                         
                                prompt = f"""### Instruction:
                                            Use the context below to answer the question clearly.
                                            ### Context: {context}
                                            ### Question:{question}
                                            # ### Answer:"""

                                response = generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.3)
                                answer = response[0]["generated_text"].split("### Answer:")[-1].strip()
                            else:
                                answer = "No relevant information found in the document."

                        except Exception as e:
                            answer = f"Error: {str(e)}"

                        results.append(answer)


                    df["Clarification Response"] = results
                    st.success("All queries processed successfully.")
                    st.dataframe(df[["Query Raised", "Clarification Response"]])

                    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx").name
                    df.to_excel(output_path, index=False)
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="Download Excel",
                            data=f,
                            file_name="clarification_responses.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
        except Exception as e:
            st.error(f"Failed to process Excel file: {e}")
