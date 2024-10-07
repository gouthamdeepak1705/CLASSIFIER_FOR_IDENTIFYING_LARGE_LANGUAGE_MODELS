import pandas as pd
import torch
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


file_path = '2000.xlsx'
output_file_path = 'EnglishTenseTruncatedDataset_Updated_FlanT5_Detailed_GPU_Batch_progress.xlsx'
temp_save_path = 'temp_progress_checkpoint.xlsx'

df = pd.read_excel(file_path)

if os.path.exists(temp_save_path):
    df_checkpoint = pd.read_excel(temp_save_path)
    completed_indexes = df_checkpoint.index.tolist()
    print(f"Resuming from previously saved progress with {len(completed_indexes)} sentences already completed.")
else:
    df_checkpoint = pd.DataFrame(columns=['Truncated_Sentence', 'Generated_xj', 'Generated_complete'])
    completed_indexes = []


tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base').to(device)

def complete_sentence(truncated_sentence, min_word_count=10, max_retries=30):
    prompt = f"Complete this sentence in as much detail as possible: {truncated_sentence}"
    
    for attempt in range(max_retries):
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
            outputs = model.generate(inputs["input_ids"], max_length=150, num_return_sequences=1,
                                     do_sample=True, top_k=50, top_p=0.95)

            completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

            for end_marker in ['.', '!', '?']:
                if end_marker in completion:
                    completion = completion.split(end_marker)[0] + end_marker
                    break

            word_count = len(completion.split())
            if word_count >= min_word_count:
                return completion.strip()

        except Exception as e:
            print(f"Error generating completion for sentence: {truncated_sentence}")
            print(f"Error: {e}")
    
    return completion.strip() if completion else "[Error generating completion]"

batch_size = 64
completion_list = []
combined_list = []

for i in tqdm(range(0, len(df), batch_size)):
    if i in completed_indexes:
        continue  

    batch = df['Truncated_Sentence'][i:i + batch_size].tolist()
    
    for sentence in batch:
        completion = complete_sentence(sentence, min_word_count=10)
        completion_list.append(completion)
        combined_list.append(f"{sentence} {completion}")

    df_temp = pd.DataFrame({'Truncated_Sentence': batch,
                            'Generated_xj': completion_list[-batch_size:], 
                            'Generated_complete': combined_list[-batch_size:]})
    
    df_checkpoint = pd.concat([df_checkpoint, df_temp])
    
    df_checkpoint.to_excel(temp_save_path, index=False)

    completion_list.clear()
    combined_list.clear()

df_checkpoint.to_excel(output_file_path, index=False)
print(f"Excel file updated and saved to: {output_file_path}")

if os.path.exists(temp_save_path):
    os.remove(temp_save_path)
    print(f"Temporary checkpoint file {temp_save_path} has been deleted.")