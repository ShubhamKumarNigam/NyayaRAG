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
        summarized_text = row["summarized_text"]    ### MENTION THE COLUMN NAME WITH INPUT DATA ###
        cited_cases_merged = "\n\n".join(
            f"**{title}**\n{row['cited_cases_data'][key]}"  ### MENTION THE COLUMN NAME WITH CITED CASES DATA ###   
            for title, key in zip(row["cited_cases"], row["cited_cases_data"])
        )

        prompt = f"""
            You are a legal expert tasked with making a judgment about whether an appeal should be accepted or rejected based on the provided summary of the case and relevant cited cases. Your task is to evaluate whether the appeal should be accepted (1) or rejected (0) based on the legal principles and precedents provided below.

            ### Example 1:
            Case proceedings: "The appellant was denied pension benefits under Rule 30 of the CCS (Pension) Rules, 1977. The authorities rejected his request, stating that his service conditions did not fulfill the necessary criteria."

            Relevant Cited Cases:
            **Government Of Andhra Pradesh & Another vs Dr. R. Murali Babu Rao & Anr.**  
            - The court ruled that eligibility for benefits under a special rule requires explicit inclusion in the appointment conditions.  
            **A.N. Shashtri vs State Of Punjab & Ors**  
            - The court held that mere length of service does not automatically entitle an employee to pension benefits.  
            **XYZ Case**  
            - Similar pension claims were rejected due to lack of specific eligibility criteria in service rules.  

            ##PREDICTION: 0  
            ##EXPLANATION: The appeal is rejected because the appellant's service conditions did not explicitly fulfill Rule 30’s requirements, as established in past rulings.  

            ### Example 2:
            Case proceedings: "The petitioner, an employee of a government research institute, challenged the denial of promotion, arguing that he met all eligibility criteria under the organization's service rules."

            Relevant Cited Cases:
            **ABC vs Union of India**  
            - The court ruled that promotions cannot be denied arbitrarily if eligibility criteria are met.  
            **DEF vs State of Karnataka**  
            - Established that performance and eligibility should take precedence over seniority unless stated otherwise in rules.  
            **GHI vs Public Service Commission**  
            - Held that the selection process must be transparent and based on merit.  

            ##PREDICTION: 1  
            ##EXPLANATION: The appeal is accepted because the petitioner met all eligibility criteria, and the selection committee’s decision lacked a strong legal basis, making the denial arbitrary.  

            ### Now, evaluate the following case:
            
            Case proceedings: {summarized_text}

            Relevant Cited Cases:
            {cited_cases_merged}
            
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
