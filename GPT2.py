from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

model.config.pad_token_id = model.config.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

hyperparameters = {
    'max_length': 150,
    'temperature': 0.7, 
    'top_k': 50,
    'top_p': 0.9,
    'no_repeat_ngram_size': 2,
    'do_sample': True
}

def generate_until_full_stop_gpt2(truncated_sentence, model, tokenizer, params=hyperparameters, min_word_count=10, max_retries=20):
    prompt = truncated_sentence

    for attempt in range(max_retries):
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + params['max_length'],
                temperature=params['temperature'],
                top_k=params['top_k'],
                top_p=params['top_p'],
                no_repeat_ngram_size=params['no_repeat_ngram_size'],
                do_sample=params['do_sample'],
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        completion = generated_text[len(truncated_sentence):].strip()

        for end_marker in ['.', '!', '?']:
            if end_marker in completion:
                completion = completion.split(end_marker)[0] + end_marker
                break

        word_count = len(completion.split())
        if word_count >= min_word_count:
            return completion.strip()

    return completion.strip() if completion else "[Error generating completion]"

file_path = "2000.xlsx"
output_file_path = "EnglishTenseWithGPT2Completions_GPU_Progress.xlsx"
temp_save_path = "GPT2_checkpoint.xlsx"

df = pd.read_excel(file_path)

if os.path.exists(temp_save_path):
    df_checkpoint = pd.read_excel(temp_save_path)
    completed_indexes = df_checkpoint.index.tolist()
    print(f"Resuming from {len(completed_indexes)} completed sentences.")
else:
    df_checkpoint = pd.DataFrame(columns=['Truncated_Sentence', 'GPT2_xj', 'GPT2_complete'])
    completed_indexes = []

batch_size = 5
completion_list = []
combined_list = []

total_batches = len(range(0, len(df), batch_size))
for batch_num, i in enumerate(range(0, len(df), batch_size)):
    if i in completed_indexes:
        continue
    
    batch = df['Truncated_Sentence'][i:i + batch_size].tolist()
    
    for sentence in batch:
        completion = generate_until_full_stop_gpt2(sentence, model, tokenizer, min_word_count=10)
        completion_list.append(completion)
        combined_list.append(f"{sentence} {completion}")

    df_temp = pd.DataFrame({'Truncated_Sentence': batch,
                            'GPT2_xj': completion_list[-batch_size:],
                            'GPT2_complete': combined_list[-batch_size:]})
    
    df_checkpoint = pd.concat([df_checkpoint, df_temp])

    df_checkpoint.to_excel(temp_save_path, index=False)

    completion_list.clear()
    combined_list.clear()

    print(f"Batch {batch_num + 1}/{total_batches} completed.")

df_checkpoint.to_excel(output_file_path, index=False)
print(f"Excel file updated and saved to: {output_file_path}")

if os.path.exists(temp_save_path):
    os.remove(temp_save_path)
    print(f"Temporary checkpoint file {temp_save_path} has been deleted.")