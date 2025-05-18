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
model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto") #.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the pad_token manually if not defined
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token

# Open the output CSV file in append mode
output_csv_file = ""  ### SPECIFY THE OUTPUT CSV FILE NAME AND PATH ###
with open(output_csv_file, 'a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(list(df.columns) + ["llama3.1_8B_pred"])  # Write the header

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        summarized_text = row["summarized_text"]
        facts = row["facts"]
        sections = row["sections"]
        cited_cases = row["cited_cases"]
        cited_cases_data = row["cited_cases_data"]
        
        cited_cases_merged = "\n\n".join(
            f"**{title}** - {cited_cases_data[key]}"
            for title, key in zip(cited_cases, cited_cases_data)
        )

        statutes_merged = "\n\n".join(
            f"**{section}** - {value}" for section, value in sections.items()
        )
        prompt = f"""
        You are a legal expert tasked with making a judgment about whether an appeal should be accepted or rejected based on the provided case details. Your task is to evaluate whether the appeal should be accepted (1) or rejected (0) solely based on the following legal aspects:

        1. Summary of Case Proceedings: A concise overview of the case proceedings, including arguments, court observations, and key developments.  
        2. Cited Cases: Relevant past judgments cited in this case.  
        3. Relevant Sections: Applicable laws or legal provisions referenced.  

        ### Example 1:
        Summary of Case Proceedings: "The petitioner challenged the termination of employment, arguing that no prior notice was provided despite a contractual obligation. The respondent (employer) contended that termination was due to misconduct, but no official warning or evidence was presented. The lower court found in favor of the petitioner, ruling that termination without notice was unjustified under labor laws."  

        Cited Cases: "Ram Singh v. XYZ Ltd. (2020) – Ruled that termination without notice violates contractual obligations unless justified."  

        Relevant Sections: "Section 25F of the Industrial Disputes Act – Requires notice or compensation for termination."  

        ##PREDICTION: 1  
        ##EXPLANATION: The appeal is accepted because the employer failed to justify termination without notice. The cited case establishes precedent that unjustified termination breaches contractual rights, further supported by statutory law.  

        ---

        ### Example 2:
        Summary of Case Proceedings: "The plaintiff alleged medical negligence following a surgical procedure, claiming inadequate post-operative care. However, hospital records confirmed that the patient was informed of all potential risks and signed a consent form. The defendant (hospital) provided expert testimony and medical records showing adherence to standard protocols. The lower court dismissed the negligence claim, citing lack of evidence of deviation from standard care."  

        Cited Cases: "ABC Hospital v. John Doe (2015) – Held that informed consent negates liability unless negligence is proven."  

        Relevant Sections: "Consumer Protection Act, Section 2(1)(g) – Defines medical negligence based on deviation from standard care."  

        ##PREDICTION: 0  
        ##EXPLANATION: The appeal is rejected because the plaintiff was informed about potential risks before the surgery. The cited case confirms that signing an informed consent form limits negligence claims unless there is clear deviation from medical standards, which was not found here.  

        ---

        ### Now, evaluate the following case:

        Summary of Case Proceedings: {summarized_text}  

        Cited Cases: {cited_cases_merged if cited_cases else "None"}  

        Relevant Sections: {statutes_merged if sections else "None"}  

        Provide your judgment by strictly following this format:

           ##PREDICTION: [Insert your prediction here]  
           ##EXPLANATION: [Insert your reasoning here that led you to your prediction.]

        Strictly do not include anything outside this format. Strictly follow the provided format. Do not generate placeholders like [Insert your prediction here]. Just provide the final judgment and explanation within 750 tokens. Do not hallucinate/repeat the same sentence again and again.
        """
        
        # Tokenize and run inference with the base model
        input_ids = tokenizer(prompt, return_tensors='pt', truncation=True, padding=True).input_ids.to(device)
        outputs = model.generate(input_ids=input_ids, max_new_tokens=750, pad_token_id=tokenizer.eos_token_id)
        
        # Decode the output
        output = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
        
        # Write the results to the CSV file
        writer.writerow(list(row) + [output])
        print(output)
