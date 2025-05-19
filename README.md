# NyayaRAG: Realistic Legal Judgment Prediction with RAG under the Indian Common Law System

**NyayaRAG** is a Retrieval-Augmented Generation (RAG) framework designed for **legal judgment prediction and explanation** in the Indian common law system. It integrates **factual case descriptions**, **statutory provisions**, and **semantically retrieved prior cases** to emulate real-world courtroom reasoning.

---

## 🧠 Motivation

Legal Judgment Prediction (LJP) in India traditionally ignores a vital aspect of the judicial process: reliance on **statutes and precedent**. NyayaRAG bridges this gap by simulating how Indian judges make decisions using:

- Factual case summaries
- Relevant legal statutes
- Prior similar judgments (explicit and implicit)

This improves both **predictive accuracy** and **explanation quality**, ensuring decisions are grounded in realistic and verifiable legal contexts.

---

## 🔍 Key Contributions

- **Realistic LJP Framework**: We propose a novel RAG-based setup aligned with Indian legal reasoning using facts, laws, and precedents.
- **Modular Input Pipelines**: Various configurations combining case text, statutes, cited precedents, and semantically retrieved similar cases.
- **Grounded Explanations**: Legal explanations are generated that mirror actual judicial opinion writing.
- **Public Dataset**: A structured corpus of 56k+ Supreme Court judgments with modular summaries and metadata.

---

## 📊 Dataset

We collected and processed **56,387** Supreme Court judgments from [IndianKanoon.org](https://indiankanoon.org). Key components:

| Component              | Count / Notes                                             |
|------------------------|-----------------------------------------------------------|
| Full Judgments         | 56,387 documents                                          |
| Summarized Cases       | ~5,000 curated with LLMs (700–1000 tokens)                |
| Facts Only             | Factual narratives extracted separately                   |
| Statutes               | Auto-extracted legal provisions (e.g., IPC, Constitution) |
| Precedents             | Cited cases and semantically retrieved top-k judgments    |

All documents are available in a format suitable for RAG-based pipelines.

---

## 🧪 Methodology

### ⚙️ RAG Pipeline Design

We define several input pipelines:
- `CaseText Only`
- `CaseText + Statutes`
- `CaseText + Precedents`
- `CaseText + Previous Similar Cases`
- `CaseText + Statutes + Precedents`
- `Facts Only`
- `Facts + Statutes + Precedents`

These combinations allow granular control to evaluate how each input type affects performance.

### 🔧 Inference Setup
- **LLM**: LLaMA-3.1 8B Instruct (few-shot prompting)
- **Summarization**: Mixtral-8x7B-Instruct-v0.1 (used to condense inputs)
- **Retrieval**: ChromaDB with MiniLM-L6-v2 embeddings

---

## 📈 Evaluation

We evaluated both **prediction** and **explanation quality** using:

- **Classification Metrics**: Accuracy, F1, Precision, Recall
- **Lexical Metrics**: ROUGE, BLEU, METEOR
- **Semantic Metrics**: BERTScore, BLANC
- **LLM-based Evaluation**: G-Eval (GPT-4o-mini)

> 🏆 Best-performing pipeline: `CaseText + Statutes` for both judgment prediction and explanation generation.


---

## ⚖️ Motivation

India's legal system is overburdened with millions of pending cases. Most existing Legal Judgment Prediction (LJP) systems rely solely on case text, ignoring crucial legal reasoning based on **statutes and judicial precedents** — a fundamental part of Indian jurisprudence. **NyayaRAG** bridges this gap by grounding predictions in external legal knowledge.

---

## 🧱 Key Features

- **Realistic Legal Inputs:** Models receive not just factual case summaries but also applicable laws and top-k similar past cases.
- **Retrieval-Augmented Generation (RAG):** Prevents hallucination and ensures factual legal grounding.
- **Explanation Generation:** Predicts binary verdicts (accept/reject) along with coherent legal justifications.
- **Evaluation with G-Eval (LLM-as-a-Judge):** Beyond lexical metrics, we use GPT-4-based evaluation for legal soundness.

---

## 📊 Dataset

- **Source:** 56,387 Supreme Court of India judgments (from [IndianKanoon](https://indiankanoon.org))
- **Summarized using:** Mixtral-8x7B-Instruct
- **Components:**
  - Factual narratives
  - Statutory references (e.g., IPC, Constitution)
  - Explicitly cited precedents
  - Semantically similar prior cases via ChromaDB
- **Vector Embeddings:** `all-MiniLM-L6-v2`
- **Retrieval:** Top-3 most similar cases per input

---

## 🔧 Methodology

1. **Summarization:** Long judgments are shortened using a tailored prompt to extract key elements.
2. **Retrieval Pipelines:** We construct multiple input pipelines:
   - `Facts Only`
   - `CaseText Only`
   - `CaseText + Statutes`
   - `CaseText + Precedents`
   - `CaseText + Previous Similar Cases`
   - `CaseText + Statutes + Precedents`
   - `Facts + Statutes + Precedents`
3. **Prediction & Explanation:**
   - Binary Decision: 0 (Rejected) / 1 (Accepted)
   - Legal Explanation: Natural language output referencing laws and precedents
4. **Model:** LLaMA 3–8B Instruct with few-shot prompting

---

## 📈 Evaluation

### 🔹 Judgment Prediction
- Metrics: Accuracy, Precision, Recall, F1-Score
- Best performance: `CaseText + Statutes` pipeline

### 🔹 Explanation Generation
- Metrics: ROUGE, BLEU, METEOR, BERTScore, BLANC
- G-Eval (LLM-based): Legal soundness, factuality, and clarity
- Highest G-Eval Score: `CaseText + Statutes` (4.21/10)

---

## 📌 Key Results

| Pipeline                           | Accuracy (%) | G-Eval Score |
|------------------------------------|--------------|--------------|
| CaseText Only                      | 62.27        | 4.17         |
| CaseText + Statutes               | **67.07**    | **4.21**     |
| CaseText + Statutes + Precedents  | 64.70        | 4.11         |
| Facts Only                         | 51.13        | 3.53         |

---
## 🔍 Future Work

NyayaRAG opens several promising avenues for future enhancements:

- **Multi-class or Hierarchical Verdicts:** Extend beyond binary outcomes to better reflect the complexity of real-world legal decisions.
- **Symbolic and Graph-based Legal Reasoning:** Combine dense retrieval with structured knowledge from legal ontologies and statute graphs.
- **Temporal Precedent Modeling:** Incorporate the time dimension to differentiate between outdated and recent precedents.
- **Human-in-the-loop Validation:** Enable feedback from legal professionals to improve reliability and trust.
- **Cross-jurisdictional Adaptation:** Explore adaptation of the framework to other common law systems beyond India.

---

## 📜 Disclaimer & Ethical Considerations

- The legal documents used were sourced from [IndianKanoon](https://indiankanoon.org), a public legal repository.
- This system is meant **strictly for research and academic purposes** and **not for real-world deployment**.
- No private or sensitive data has been used. All processing was performed on public data.
- Outputs from this model **do not constitute legal advice** and must not be used for legal decision-making without expert oversight.
- Although the system attempts to replicate human-like legal reasoning, it may still reflect biases present in source judgments or training data.


