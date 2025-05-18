
## 🧪 Overview of the code Pipelines

This repository contains **7 pipelines** designed to process different aspects of **Indian Supreme Court case law** for **structured legal judgment prediction**. Each pipeline works with varying data formats (e.g., **single document**, **multi-document**) and generates legal judgment predictions, leveraging **pre-trained language models** such as **Llama-3.1-8B-Instruct**.

Each pipeline processes a specific combination of legal data (e.g., **summarized case text**, **statutes**, **cited cases**, **facts**), performs relevant operations, and generates judgment predictions in the form of **acceptance (1)** or **rejection (0)**, along with an **explanation** for the decision.

---

### 📁 Common Pipeline Workflow

Regardless of the specific pipeline being run, each follows a similar general workflow:

1. **Data Loading**  
   - Input data is loaded from a **JSON file** containing relevant legal information, including:
     - Summarized case text
     - Relevant statutes
     - Cited cases
     - Case facts (if applicable)

2. **Data Preprocessing**  
   - The input data is processed to extract relevant sections, ensuring that the text is within the allowable token limit for model inference. This might include:
     - Preprocessing case summaries, statutes, and facts into manageable text chunks.
     - Handling multiple documents where applicable (i.e., for multi-document inputs).

3. **Model Loading**  
   - A **pre-trained language model** (e.g., **Llama-3.1-8B-Instruct**) is loaded using the **Hugging Face** `transformers` library.
   - The model is set to **CUDA** for faster computation if available, or defaults to **CPU**.

4. **Inference**  
   - A **prompt** is created that combines the legal case data and relevant statutes to generate a judgment prediction. This is fed into the model to produce a predicted outcome (either **accept** or **reject**) and an **explanation**.

5. **Result Generation**  
   - The prediction and explanation are written to an **output CSV file** for further analysis. Each row in the output CSV corresponds to a single case, with the predicted outcome and associated reasoning.

---

### 📁 Pipeline Script Structure

Each pipeline follows the same general structure with minor modifications based on the specific data inputs:

- **Data Loading**: Load the relevant JSON dataset for the pipeline.
- **Preprocessing**: Prepare the input text to fit model requirements (token length, format, etc.).
- **Model Loading**: Load a specific model for legal judgment prediction (e.g., Llama-3.1-8B-Instruct).
- **Model Inference**: Run inference on the data, generating predictions and explanations.
- **Output**: Save the predictions and explanations to a CSV file.

---
