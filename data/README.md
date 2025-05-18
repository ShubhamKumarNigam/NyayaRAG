
# 🇮🇳📚 पुनर्प्राप्ति-संवर्धित विधि प्रणाली (RAG-based Legal Framework)

A **Retrieval-Augmented Generation (RAG)** framework for **structured legal judgment prediction** in the **Indian judicial domain**, built using the **Mixtral 8x7B Instruct-v0.1** model.

---

## 🧠 Overview

This project focuses on building a structured pipeline to work with **Supreme Court of India** case law, leveraging summarization and retrieval using state-of-the-art large language models. It organizes legal data into meaningful sections and enables inference over both **single** and **multi-document** inputs.

---

## 🗂️ Directory Structure & Contents

Each folder in this repo contains data specific to various stages and components of a legal judgment. Below is a high-level description of each:

### 📁 `1. Base_dataset/`
Contains raw and pre-processed case data used across all pipelines.

| File | Description |
|------|-------------|
| `SCI_judgements_56k.json` | Scraped from IndianKanoon.org, includes cited statutes. |
| `updated_SCI_56k_(multi/single).json` | Sectioned into facts, statutes, and final judgment (0/1 label). |
| `SCI_56k_(single/multi)_5k_summarized.json` | Summarized version of full cases, shortened for LLM compatibility. |

---

### 📁 `2. CaseText_only/`
Houses summarized case texts without additional metadata.

| File | Description |
|------|-------------|
| `SCI_56k_(single/multi)_5k_summarized.json` | Directly sourced summaries from the base dataset. |

---

### 📁 `3. CaseText_Statutes/`
Contains summaries of case texts along with **cited statutes**.

| File | Description |
|------|-------------|
| `SCI_56k_(single/multi)_5k_summarised_w_sections.json` | Includes case summaries + extracted statutes. |

---

### 📁 `4. CaseText_Cited/`
Includes both raw and summarized case data with **cited cases**.

| File | Description |
|------|-------------|
| `CasePlusCitedCases_(single/multi).json` | Raw data with cited cases from IndianKanoon.org. |
| `CasePlusCitedCases_(single/multi)_summarized.json` | Summarized using Mixtral model. |

---

### 📁 `5. CaseText_Previous_Similar_Cases/`
Focused on similar precedent cases.

| File | Description |
|------|-------------|
| `SCI_judgements_56k_summarized.json` | Summarized judgments from base data. |
| `SCI_56k_(single/multi)_5k_summarized.json` | Same as above, filtered by split. |

---

### 📁 `6. Facts_only/`
Extracts and summarizes only the **facts and judgment**.

| File | Description |
|------|-------------|
| `5k_(single/multi)_summarized_facts.json` | Contains summarized facts and case outcomes. |

---

### 📁 `7. Facts_Statutes_Cited/`
All-in-one enriched dataset — **facts**, **statutes**, and **cited cases**.

| File | Description |
|------|-------------|
| `5k_(single/multi)_summarised_CitedPlusFacts.json` | Most information-rich summaries for inference. |

---

## 🔎 Case Judgment Components

Before using the pipeline, familiarize yourself with the components extracted from each legal case:

- **🧾 Statutes**: Legal provisions cited in the case (aka "cited statutes").
- **📚 Cited Cases**: Previous judgments referenced for legal reasoning.
- **🔁 Previous Similar Cases**: Judgments with similar factual/legal context.
- **📖 Facts**: Actual details of the case (events, evidence, context).

---

## 🧪 Inference Scripts

Run inference for either **single** or **multi-document** input formats using:

```bash
python inference_single.py   # For single-file datasets
python inference_multi.py    # For multi-file datasets
