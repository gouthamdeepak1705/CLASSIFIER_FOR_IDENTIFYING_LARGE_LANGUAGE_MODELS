!pip install transformers torch pandas tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_name = 'facebook/opt-1.3b'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model.to(device)

hyperparameters = {
    'max_length': 150,
    'temperature': 0.7,
    'top_k': 50,
    'top_p': 0.9,
    'do_sample': True,
    'no_repeat_ngram_size': 2,
    'early_stopping': True
}

def generate_completion_opt(truncated_sentence, model, tokenizer, params=hyperparameters, min_word_count=10, max_retries=30):
    for attempt in range(max_retries):
        input_ids = tokenizer.encode(truncated_sentence, return_tensors='pt').to(device)

        output_ids = model.generate(
            input_ids=input_ids,
            max_length=input_ids.shape[1] + params['max_length'],
            temperature=params['temperature'],
            top_k=params['top_k'],
            top_p=params['top_p'],
            do_sample=params['do_sample'],
            no_repeat_ngram_size=params['no_repeat_ngram_size'],
            early_stopping=params['early_stopping'],
            pad_token_id=tokenizer.eos_token_id
        )

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        completion = generated_text[len(truncated_sentence):].strip()

        for end_marker in ['.', '!', '?']:
            if end_marker in completion:
                completion = completion.split(end_marker)[0] + end_marker
                break

        if len(completion.split()) >= min_word_count:
            return completion

    return completion

file_path = "2000.xlsx"
output_file_path = "EnglishTenseWithOPTCompletions_GPU_Progress.xlsx"
temp_save_path = "OPT_checkpoint.xlsx"

df = pd.read_excel(file_path)

if os.path.exists(temp_save_path):
    df_checkpoint = pd.read_excel(temp_save_path)
    completed_indexes = df_checkpoint.index.tolist()
    print(f"Resuming from {len(completed_indexes)} completed sentences.")
else:
    df_checkpoint = pd.DataFrame(columns=['Truncated_Sentence', 'OPT_xj', 'OPT_complete'])
    completed_indexes = []

batch_size = 1
completion_list = []
combined_list = []

total_batches = len(range(0, len(df), batch_size))
for batch_num, i in enumerate(range(0, len(df), batch_size)):
    if i in completed_indexes:
        continue
    
    batch = df['Truncated_Sentence'][i:i + batch_size].tolist()
    batch_completions = [generate_completion_opt(sentence, model, tokenizer, min_word_count=10) for sentence in batch]
    
    combined = [f"{sentence} {completion}" for sentence, completion in zip(batch, batch_completions)]
    
    df_temp = pd.DataFrame({'Truncated_Sentence': batch,
                            'OPT_xj': batch_completions,
                            'OPT_complete': combined})
    
    df_checkpoint = pd.concat([df_checkpoint, df_temp])

    df_checkpoint.to_excel(temp_save_path, index=False)
    print(f"Batch {batch_num + 1}/{total_batches} completed.")

df_checkpoint.to_excel(output_file_path, index=False)
print(f"Excel file updated and saved to: {output_file_path}")

if os.path.exists(temp_save_path):
    os.remove(temp_save_path)
    print(f"Temporary checkpoint file {temp_save_path} has been deleted.")