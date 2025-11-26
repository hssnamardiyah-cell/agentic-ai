import streamlit as st
import os
import torch
import joblib
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Any, List, Dict

# Konfigurasi Halaman
st.set_page_config(page_title="Agentic AI Molecule Generator", layout="wide")

# Cek RDKit
try:
    from rdkit import Chem
except ImportError:
    st.error("RDKit tidak ditemukan. Pastikan Dockerfile benar.")
    Chem = None

# --- DEFINISI MODEL (Sesuai Notebook Anda) ---
class ChemBERTaMulti(torch.nn.Module):
    def __init__(self, n_outputs):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        self.head = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, n_outputs)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.head(pooled)

# --- FUNGSI LOAD MODEL (Cached) ---
@st.cache_resource
def load_resources():
    base_path = "models" # Folder models di server
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # 1. Load T5
        t5_path = os.path.join(base_path, "molT5_finetuned_fixed")
        tokenizer_t5 = T5Tokenizer.from_pretrained(t5_path)
        model_t5 = T5ForConditionalGeneration.from_pretrained(t5_path).to(device)
        
        # 2. Load ChemBERTa
        bert_path = os.path.join(base_path, "chemberta_multi_model.pth")
        tok_path = os.path.join(base_path, "chemberta_multi_tokenizer")
        scaler_path = os.path.join(base_path, "label_scaler.pkl")
        
        bert_tokenizer = AutoTokenizer.from_pretrained(tok_path)
        label_scaler = joblib.load(scaler_path)
        
        bert_model = ChemBERTaMulti(n_outputs=5)
        # Load state dict dengan map_location cpu (jika server tidak ada GPU)
        bert_model.load_state_dict(torch.load(bert_path, map_location=device))
        bert_model.to(device)
        bert_model.eval()
        
        return tokenizer_t5, model_t5, bert_model, bert_tokenizer, label_scaler, device
    except Exception as e:
        st.error(f"Gagal memuat model. Error: {e}")
        return None

# --- LOGIKA GENERASI (Disederhanakan untuk Demo) ---
def run_pipeline(constraints, api_key, resources):
    tokenizer_t5, model_t5, bert_model, bert_tokenizer, scaler, device = resources
    
    # 1. Generate dengan T5
    caption_parts = [f"{k}={v}" for k, v in constraints.items()]
    caption = "properties: " + ", ".join(caption_parts)
    
    inputs = tokenizer_t5(caption, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model_t5.generate(inputs, max_length=128, num_return_sequences=3, do_sample=True)
    
    smiles_list = [tokenizer_t5.decode(o, skip_special_tokens=True) for o in outputs]
    
    # 2. Prediksi Properti dengan ChemBERTa
    results = []
    for s in smiles_list:
        if Chem and Chem.MolFromSmiles(s): # Validasi Kimia
            enc = bert_tokenizer(s, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                pred = bert_model(enc['input_ids'], enc['attention_mask']).cpu().numpy()
            
            vals = scaler.inverse_transform(pred)[0]
            results.append({
                "SMILES": s, 
                "mu": vals[0], "alpha": vals[1], "gap": vals[2], "Cv": vals[3], "atoms": vals[4]
            })
            
    return results

# --- TAMPILAN UI ---
st.title("🧪 Agentic AI: Molecule Generator")

# Load Resources
resources = load_resources()

if resources:
    st.success("Model AI Siap!")
    
    col1, col2 = st.columns(2)
    with col1:
        mu = st.number_input("Target Mu", 0.0)
        alpha = st.number_input("Target Alpha", 13.21)
    with col2:
        gap = st.number_input("Target Gap", 13.73)
        cv = st.number_input("Target Cv", 6.47)
        
    api_key = st.text_input("OpenAI API Key (Opsional untuk Agent LangGraph)", type="password")
    
    if st.button("Generate Molekul"):
        with st.spinner("Sedang memproses..."):
            constraints = {"mu": mu, "alpha": alpha, "gap": gap, "Cv": cv}
            results = run_pipeline(constraints, api_key, resources)
            
            if results:
                st.dataframe(pd.DataFrame(results))
            else:
                st.warning("Tidak ada molekul valid yang dihasilkan.")
else:
    st.warning("Silakan upload file model ke folder 'models/' sebelum deploy.")