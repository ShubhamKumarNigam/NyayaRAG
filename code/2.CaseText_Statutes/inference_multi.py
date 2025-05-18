import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import csv
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

CUSTOM_CACHE_DIR = ""   ### MENTION CACHE DIRECTORY HERE IF YOU WANT. DEFAULT IS HOME ###
os.environ["HF_HOME"] = CUSTOM_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = CUSTOM_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CUSTOM_CACHE_DIR

# Load the data from a JSON file
input_json_file = ""  ### SPECIFY THE PATH TO YOUR INPUT JSON FILE ###
with open(input_json_file, 'r') as file:
    data = json.load(file)

# Convert the JSON data into a pandas DataFrame (this assumes each entry is a dictionary)
df = pd.DataFrame(data)

# Functions to preprocess the input and output 
def preprocess_input(text):
    max_tokens = 3000  # Adjust according to max tokens you need from Input Case description
    tokens = text.split(' ')
    num_tokens_to_extract = min(max_tokens, len(tokens))
    text1 = ' '.join(tokens[-num_tokens_to_extract:len(tokens)])
    return text1

def preprocess_output(text):
    max_tokens = 500  # Adjust according to max tokens you need from Official Reasoning
    tokens = text.split(' ')
    num_tokens_to_extract = min(max_tokens, len(tokens))
    text1 = ' '.join(tokens[-num_tokens_to_extract:len(tokens)])
    return text1

# Preprocess the input cases
#for i, row in tqdm(df.iterrows(), total=df.shape[0]):
 #   inp = row['summarized_text']
  #  inpu = preprocess_input(inp)
   # df.at[i, 'summarized_text'] = inpu

#for i, row in tqdm(df.iterrows(), total=df.shape[0]):
 #   inp = row['facts']
  #  inpu = preprocess_input(inp)
   # df.at[i, 'facts'] = inpu
    
# Load the base model and tokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Specify the model name
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the pad_token manually if not defined
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token

# Open the output CSV file in append mode
output_csv_file = ""  ### SPECIFY THE OUTPUT CSV FILE NAME AND PATH ###
with open(output_csv_file, 'a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(list(df.columns) + ["llama3.1_8B_pred"])  # Write the header

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        summarized_text = row["summarized_text"]   ### MENTION THE COLUMN NAME WITH INPUT DATA ###
        relevant_statues = row["sections"]  ### MENTION THE COLUMN NAME WITH STATUTES DATA ###
        prompt = f"""
        You are a legal expert tasked with making a judgment about whether an appeal should be accepted or rejected based on the provided summary of the case and relevant statutes. Your task is to evaluate whether the appeal should be accepted (1) or rejected (0) based on the case proceedings and the applicable legal provisions provided below.

        ### Example 1:
        Case proceedings: "The defendant was convicted for assaulting a police officer during a protest. The incident was recorded on CCTV footage."
        Relevant statutes: "Section 96 of the Indian Penal Code (IPC) states that an act done in self-defense is not an offense if there is a reasonable apprehension of harm."
        ##PREDICTION: 0
        ##EXPLANATION: The appeal is rejected because while self-defense is a valid legal argument under Section 96 IPC, there is insufficient evidence to prove that the defendant's response was proportional or necessary to prevent harm.

        ### Example 2:
        Case proceedings: "The defendant was accused of embezzling funds from a charity. An audit revealed the misappropriation of large sums of money."
        Relevant statutes: "Section 405 of the IPC defines criminal breach of trust as dishonest misappropriation or conversion of entrusted property."
        ##PREDICTION: 1
        ##EXPLANATION: The appeal is accepted because, under Section 405 IPC, failing to account for entrusted funds constitutes a breach of trust, and the defendant could not provide a legitimate explanation for the missing money.

        ### Now, evaluate the following case:

        Case proceedings: {summarized_text}

        Relevant statutes: {relevant_statutes}

        Provide your judgment by strictly following this format:

           ##PREDICTION: [Insert your prediction here]
           ##EXPLANATION: [Insert your reasoning here that led you to your prediction.]
           
        Strictly do not include anything outside this format. Strictly follow the provided format. Do not generate placeholders like [Insert your prediction here]. Just provide the final judgment and explanation. Do not hallucinate/repeat the same sentence again and again.
        """
        
        # Tokenize and run inference with the base model
        input_ids = tokenizer(prompt, return_tensors='pt', truncation=True, padding=True).input_ids.to(device)
        outputs = model.generate(input_ids=input_ids, max_new_tokens=750, pad_token_id=tokenizer.eos_token_id)
        
        # Decode the output
        output = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
        
        # Write the results to the CSV file
        writer.writerow(list(row) + [output])
        print(output)
