Dimport json
import time
import chromadb
from huggingface_hub import InferenceClient
import csv
from tqdm import tqdm

# File paths
multi_file_path = ''   ### SPECIFY THE PATH TO YOUR INPUT JSON FILE ###
judgements_file_path = ''    ### SPECIFY THE PATH TO YOUR ALL CASES JSON FILE ###
csv_file = ''  ### SPECIFY THE PATH TO YOUR OUTPUT JSON FILE ###

# Initialize ChromaDB client
chroma_client = chromadb.Client()

collection_name = ""    ### SPECIFY THE NAME FOR THE COLLECTION ###
if collection_name not in chroma_client.list_collections():
    collection = chroma_client.create_collection(name=collection_name)
else:
    collection = chroma_client.get_collection(name=collection_name)

with open(judgements_file_path, 'r') as file:
    judgements_data = json.load(file)

for entry in tqdm(judgements_data, desc="Adding to vector database"):  
    doc_id = entry.get("document_id")
    summary = entry.get("summarized_text")  # Assuming 'summary' contains the precomputed summary

    try:
        if summary:
            collection.add(
                documents=[summary],
                metadatas=[{"source": f"source {doc_id}", "type": "summary"}],
                ids=[doc_id]
            )
    except Exception as e:
        print(f"Error adding data for Document ID {doc_id}: {e}")

# Initialize InferenceClient
client = InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct", token="") ### MENTION THE API KEY FOR HUGGINGFACE ###   

with open(multi_file_path, 'r') as file:
    multi_data = json.load(file)

for document in tqdm(multi_data, desc="Processing documents"):# Input data
    document_id = document.get('document_id')
    full_text = document.get('summarized_text', '').replace('\t', '').replace('\n', '')

    # Retrieve top 3 similar summaries from ChromaDB
    try:
        results = collection.query(
            query_texts=[full_text],
            n_results=3
        ) # Top 3 similar cases and adding in results

        similar_summaries = []
        for i, doc in enumerate(results['documents'][0]):
            similar_summaries.append(f"Similar Case Summary {i + 1}: {doc}")
    except Exception as e:
        print(f"Error retrieving similar summaries for Document ID {document_id}: {e}")
        continue

    # Combine full_text and similar summaries
    cleaned_output_texts = [text.replace('\t', '').replace('\n', '') for text in similar_summaries]
    output_text = " ".join(cleaned_output_texts)

    full_text_tokens = len(full_text.split())
    if full_text_tokens > 3000:
        remaining_tokens_needed = 3000 - len(output_text.split())
        extracted_text = full_text.split()[:remaining_tokens_needed]
        text_value = " ".join(extracted_text) + f" {'||||'} " + "\n\nThe above text is the main case. Now predict the verdict for this case. Below are summaries of similar cases for your reference to help you make your decision.\n\n" + output_text #output text is 3 similar cases together
    else:
        text_value = full_text + f" {'||||'} " + "\n\nThe above text is the main case. Now predict the verdict for this case. Below are summaries of similar cases for your reference to help you make your decision.\n\n" + output_text

    # Define the prediction query
    predict_query = "query: Predict if the appeal will be accepted (1) or dismissed (0) for the main case. Explain your decision by identifying key sentences from the document. Focus only on the main case text before the '||||' pattern. Summaries of similar cases follow this pattern for your reference to help you make your decision."
    role = 'You are an expert court judge who gives decision based on context.'
    content_with_query = f"\n{role}\n\n{predict_query}\n\n{text_value}"

    # Generate the prediction using the InferenceClient
    output2 = client.text_generation(content_with_query, max_new_tokens=500, temperature=0.2)

    # Append data to the CSV file for the current document
    csv_columns = ['document_id', 'explanation']

    # Open the CSV file in append mode
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writerow({"document_id": document_id, "explanation": output2})

    print(f"Document ID {document_id} processed.")
