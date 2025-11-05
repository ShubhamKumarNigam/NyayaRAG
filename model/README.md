# 🧠 NyayaRAG Model Collection

This repository hosts the models integrated and developed as part of the **NyayaRAG** project — a Retrieval-Augmented Generation (RAG) framework for **legal judgment prediction and explanation** under the Indian Common Law system.

NyayaRAG’s architecture combines summarization, retrieval, and generation models to emulate real-world judicial reasoning using factual case narratives, statutory laws, and judicial precedents.

---

## 1. Summarization Model

**Model:** `Mixtral-8x7B-Instruct-v0.1`  
**Task:** Legal Judgment Summarization

### Objective
Summarize lengthy Supreme Court of India judgments into structured, model-friendly texts capturing:
- Facts  
- Legal Issues  
- Statutory References  
- Reasoning
- Final Decision  

### Model Access
👉 [Mixtral-8x7B-Instruct on HuggingFace](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)

---

## 2. Retrieval Models

NyayaRAG uses retrieval augmentation to incorporate relevant statutes and prior case law, ensuring factual grounding and interpretability.

### 2.1 Embedding Model
**Model:** `all-MiniLM-L6-v2`  
**Task:** Semantic Encoding of Legal Texts

Encodes case texts, statutes, and precedents into dense vector representations for retrieval.

- Framework: SentenceTransformers  
- Embedding Dimension: 384  
- Use Case: Document indexing and query encoding for RAG

**Model Access:**  
👉 [all-MiniLM-L6-v2 on SentenceTransformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

---

### 2.2 Vector Database
**System:** `ChromaDB`  
**Task:** Semantic Retrieval of Legal Context

Stores vector embeddings and retrieves the most relevant documents (facts, statutes, prior cases) during inference.

- Retrieval Method: Cosine similarity  
- Top-k Results: 3  
- Components Retrieved: 
  - Semantically similar previous cases  

**Integration:**  
Used in NyayaRAG pipelines:  
- CaseText + Statutes  
- CaseText + Precedents  
- CaseText + Previous Similar Cases  

---

## 3. Generation Model

**Model:** `LLaMA-3.1-8B-Instruct`  
**Task:** Legal Judgment Prediction & Explanation

### Objective
Predict whether an appeal is **accepted** or **rejected** and generate a **legal explanation** grounded in retrieved statutes and precedents.

### Output Format

##PREDICTION: [0 or 1]

##EXPLANATION: [Legal reasoning and justification]

**Model Access:**  
👉 [LLaMA-3.1-8B-Instruct on Meta AI](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

---

## 4. Model Integration Summary

| Component | Model | Purpose | Notes |
|------------|--------|----------|-------|
| **Summarization** | Mixtral-8x7B-Instruct | Structured summarization of legal judgments | Long-context, domain-specific |
| **Embedding** | all-MiniLM-L6-v2 | Semantic encoding for retrieval | Lightweight, fast encoder |
| **Retrieval** | ChromaDB | Retrieve relevant statutes & precedents | Top-k (k=3) similarity-based retrieval |
| **Generation** | LLaMA-3.1-8B-Instruct | Legal judgment prediction + explanation | RAG-integrated reasoning |

---

## 5. Model Usage Pipeline

```bash
Input Judgment 
   ↓
Summarization (Mixtral)
   ↓
Embedding + Retrieval (MiniLM + ChromaDB)
   ↓
Context Augmentation
   ↓
Judgment Prediction + Explanation (LLaMA-3.1-8B)
