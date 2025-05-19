## 🧠 Overview

This project structures Supreme Court judgments into distinct legal components — such as **facts**, **cited statutes**, and **precedents** — and supports **single** and **multi-document inference** pipelines for prediction.

---

## 🗂️ Directory Structure & Contents

Each folder contains datasets for a specific component of the pipeline. Google Drive links can be used to see and download the exact data mentioned below.

---

### 📁 `1. Base_dataset/`  
[📁 Open in Google Drive](https://drive.google.com/drive/folders/1Wecj7cLxYNkjppKKRK1N_8uRy798JjJY?usp=drive_link)

| File | Description |
|------|-------------|
| `SCI_judgements_56k.json` | Scraped from IndianKanoon.org, includes cited statutes. |
| `updated_SCI_56k_single.json` | Preprocessed data with extracted sections (facts, statutes) and a binary label representing the judgment (0 = rejected, 1 = accepted). |
| `updated_SCI_56k_multi.json` | Preprocessed data with extracted sections (facts, statutes) and a binary label representing the judgment (0 = rejected, 1 = accepted). |
| `SCI_56k_single_5k_summarized.json` | Summarized version of single-document case data, reduced for efficient input into LLMs. |
| `SCI_56k_multi_5k_summarized.json` | Summarized version of multi-document case data, reduced for efficient input into LLMs. |

---

### 📁 `2. CaseText_only/`  
[📁 Open in Google Drive](https://drive.google.com/drive/folders/1p9JLQ4BRL25oKSosRErCPI2kdJDSEMxj?usp=drive_link)

| File | Description |
|------|-------------|
| `SCI_56k_single_5k_summarized.json` | Contains only the summarized text of single-document cases, directly derived from the base dataset. |
| `SCI_56k_multi_5k_summarized.json` | Contains only the summarized text of multi-document cases, directly derived from the base dataset. |

---

### 📁 `3. CaseText_Statutes/`  
[📁 Open in Google Drive](https://drive.google.com/drive/folders/1ZuvabjwwRmuoE_1nNrHb2s3uxydhsmAF?usp=drive_link)

| File | Description |
|------|-------------|
| `SCI_56k_single_5k_summarized_w_sections.json` | Summarized case text for single-document format, enriched with extracted cited statutes. |
| `SCI_56k_multi_5k_summarized_w_sections.json` | Summarized case text for multi-document format, enriched with extracted cited statutes. |

---

### 📁 `4. CaseText_Precedents/`  
[📁 Open in Google Drive](https://drive.google.com/drive/folders/1J772XCG22LeQC38esXUIRVYEuinBMO5E?usp=drive_link)

| File | Description |
|------|-------------|
| `CasePlusCitedCases_single.json` | Full single-document case data with cited case references scraped from IndianKanoon.org. |
| `CasePlusCitedCases_multi.json` | Full multi-document case data with cited case references scraped from IndianKanoon.org. |
| `CasePlusCitedCases_single_summarized.json` | Summarized version of single-document case data, including cited cases, processed with the Mixtral model. |
| `CasePlusCitedCases_multi_summarized.json` | Summarized version of multi-document case data, including cited cases, processed with the Mixtral model. |

---

### 📁 `5. CaseText_Previous_Similar_Cases/`  
[📁 Open in Google Drive](https://drive.google.com/drive/folders/1ligZDRi94ySRhA7OQU8Pq1l0Um9BHMwE?usp=drive_link)

| File | Description |
|------|-------------|
| `SCI_judgements_56k_summarized.json` | Summarized version of the full 56k dataset used for retrieving similar cases during prediction. |
| `SCI_56k_single_5k_summarized.json` | Summarized single-document case texts, used as both input and similarity reference. |
| `SCI_56k_multi_5k_summarized.json` | Summarized multi-document case texts, used as both input and similarity reference. |

---


### 📁 `6. CaseText_Statutes_Precedents/`  
[📁 Open in Google Drive](https://drive.google.com/drive/folders/14d3Adhehe6JnXoATfcwC98GmiijJ8_fq?usp=drive_link)

| File | Description |
|------|-------------|
| `CaseText_Statutes_Cited_single.json` | Summarized case text enriched with both cited statutes and cited cases for single-document format. |
| `CaseText_Statutes_Cited_multi.json` | Summarized case text enriched with both cited statutes and cited cases for multi-document format. |

---

### 📁 `7. Facts_only/`  
[📁 Open in Google Drive](https://drive.google.com/drive/folders/1YZm5PESxkF17eZbvOjy7TwpD49BGdtKx?usp=drive_link)

| File | Description |
|------|-------------|
| `5k_single_summarized_facts.json` | Summarized facts and judgments extracted from single-document cases using scraping and Mixtral summarization. |
| `5k_multi_summarized_facts.json` | Summarized facts and judgments extracted from multi-document cases using scraping and Mixtral summarization. |

---

### 📁 `8. Facts_Statutes_Precedents/`  
[📁 Open in Google Drive](https://drive.google.com/drive/folders/1lIXZRg4bi3nbcj6288jWKjZ-SdFz5RMv?usp=drive_link)

| File | Description |
|------|-------------|
| `5k_single_summarized_CitedPlusFacts.json` | Richly annotated and summarized single-document case data containing facts, cited statutes, cited cases, and judgment outcomes. |
| `5k_multi_summarized_CitedPlusFacts.json` | Richly annotated and summarized multi-document case data containing facts, cited statutes, cited cases, and judgment outcomes. |

---

## 🔎 Case Judgment Components

Each case is decomposed into structured fields:

- **🧾 Statutes**: Legal provisions cited by the judge.
- **📚 Precedents**: Past cases referenced for precedent.
- **🔁 Previous Similar Cases**: Factually or legally similar cases for consistency.
- **📖 Facts**: Key events and details of the case based on evidence.

---

## 🧪 Inference Scripts

You can run inference using either of the following scripts, depending on the input format:

```bash
python inference_single.py   # For single-file datasets
python inference_multi.py    # For multi-file datasets
