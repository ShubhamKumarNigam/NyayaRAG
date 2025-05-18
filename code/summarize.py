import os
import json
import time
from huggingface_hub import InferenceClient
from tqdm import tqdm
from transformers import AutoTokenizer

# Initialize the InferenceClient with your model and token
client = InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1", token="")    ### MENTION YOUR HUGGINGFACE API KEY HERE ###

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# File paths
input_file_path = r''  ### INPUT JSON FILE WITH CITED_CASES_DATA ###
output_file_path = r''  ### OUTPUT FILE TO STORE SUMMARIZED DATA ###

# Load the legal case judgments from JSON file
with open(input_file_path, 'r', encoding='utf-8') as file:
    legal_cases = json.load(file)

# Define model's maximum token limit (context window) and token limit for input text
MAX_TOKEN_LIMIT = 28000
INPUT_TOKEN_LIMIT = MAX_TOKEN_LIMIT - 1000  # Reserve 1000 tokens for generated summary

# Query for summarization
query = "The text is regarding a court judgement for a specific case. Summarize it into 1000 tokens but more than 700 tokens. The summarization should highlight the Facts, issues, Statute, Ratio of the decision, Ruling by Present Court (Decision) and a Conclusion"

# Function to truncate text to fit the context window with the summary token limit
def truncate_text(text, max_tokens):
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        return truncated_text
    return text

# Check if the output file already exists
processed_doc_ids = set()
if os.path.exists(output_file_path):
    with open(output_file_path, 'r', encoding='utf-8') as file:
        summarized_cases = json.load(file)
        processed_doc_ids = {case['document_id'] for case in summarized_cases}
else:
    summarized_cases = []

# Count the total number of cited cases for progress bar
total_cited_cases = sum(len(case.get('cited_cases_data', {})) for case in legal_cases if case['document_id'] not in processed_doc_ids)
print(f"Total cited cases to be summarized: {total_cited_cases}")

# Use tqdm to create a progress bar for the cited cases
with tqdm(total=total_cited_cases, desc="Processing cited cases", unit="case") as pbar:
    for case in legal_cases:
        try:
            # Ensure the case has 'document_id' and 'cited_cases_data' fields
            doc_id = case.get('document_id')
            cited_cases_data = case.get('cited_cases_data', {})

            # Skip processing if the document_id has already been summarized
            if doc_id in processed_doc_ids:
                print(f"Skipping document ID {doc_id} as it is already processed.")
                continue

            # Skip the row if cited_cases_data is empty
            if not cited_cases_data:
                print(f"Skipping document ID {doc_id} because cited_cases_data is empty.")
                continue

            # Process each cited_case in cited_cases_data
            for cited_case, case_text in cited_cases_data.items():
                # Check if case_text exceeds the token limit for input and truncate if necessary
                if len(tokenizer.encode(case_text)) > INPUT_TOKEN_LIMIT:
                    case_text = truncate_text(case_text, INPUT_TOKEN_LIMIT)
                    print(f"Document ID {doc_id} cited_case {cited_case} exceeded token limit. Truncated the text to {INPUT_TOKEN_LIMIT} tokens.")

                # Prepare content for summarization
                content_with_query = f"\n{case_text}\n\n{query}"

                # Summarize the case text
                try:
                    output = client.text_generation(content_with_query, max_new_tokens=1000, temperature=0.2)
                    time.sleep(1)
                    summarized_text = output  # Adjust this if necessary based on the actual structure of `output`
                except Exception as e:
                    if "Too Many Requests" in str(e):
                        print("Rate limit exceeded. Saving progress and sleeping for 24 hours...")
                        time.sleep(86400)  # Sleep for 24 hours
                        output = client.text_generation(content_with_query, max_new_tokens=1000, temperature=0.2)
                        summarized_text = output
                    else:
                        raise e
                print(summarized_text)
                # Replace the original case_text in cited_cases_data with the summarized text
                case['cited_cases_data'][cited_case] = summarized_text
                print(f"Document ID {doc_id} cited_case {cited_case} done.")

                # Update progress bar
                pbar.update(1)

            # Add the updated case to the summarized_cases list
            summarized_cases.append(case)

            # Save the current progress to the JSON file
            with open(output_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(summarized_cases, json_file, indent=4)  # Incrementally save the progress
        except Exception as e:
            print(f"Error summarizing document ID {doc_id}: {e}")

print(f"All summarized cases have been saved to {output_file_path}.")
