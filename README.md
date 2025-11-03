# NyayaRAG: Realistic Legal Judgment Prediction with RAG under the Indian Common Law System

**NyayaRAG** is a Retrieval-Augmented Generation (RAG) framework designed for **legal judgment prediction and explanation** in the Indian common law system. It integrates **factual case descriptions**, **statutory provisions**, and **semantically retrieved prior cases** to emulate real-world courtroom reasoning.

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
| CaseText + Statutes + Precedents  | 64.71        | 4.11         |
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


