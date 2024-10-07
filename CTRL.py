from transformers import CTRLTokenizer, CTRLLMHeadModel
import pandas as pd
import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_name = 'ctrl'
tokenizer = CTRLTokenizer.from_pretrained(model_name)
model = CTRLLMHeadModel.from_pretrained(model_name).to(device)

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

hyperparameters = {
    'max_length': 150,
    'temperature': 0.95,
    'top_k': 50,
    'top_p': 0.9,
    'do_sample': True,
    'num_return_sequences': 1
}

def generate_batch_completion_ctrl(sentences, model, tokenizer, params=hyperparameters, min_word_count=10, max_retries=30):
    control_code = 'Links'
    input_texts = [control_code + ' ' + sentence.strip() for sentence in sentences]
    
    completions = []
    
    for input_text in input_texts:
        for attempt in range(max_retries):
            input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    max_length=input_ids.shape[1] + params['max_length'],
                    temperature=params['temperature'],
                    top_k=params['top_k'],
                    top_p=params['top_p'],
                    do_sample=params['do_sample'],
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )

            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            completion = generated_text[len(input_text):].strip()

            for end_marker in ['.', '!', '?']:
                if end_marker in completion:
                    completion = completion.split(end_marker)[0] + end_marker
                    break

            if len(completion.split()) >= min_word_count:
                completions.append(completion)
                break
        else:
            completions.append(completion)

    return completions

file_path = "2000.xlsx"
output_file_path = "EnglishTenseWithCTRLCompletions_GPU_Progress.xlsx"
temp_save_path = "CTRL_checkpoint6.xlsx"

df = pd.read_excel(file_path)

if os.path.exists(temp_save_path):
    df_checkpoint = pd.read_excel(temp_save_path)
    completed_indexes = df_checkpoint.index.tolist()
    print(f"Resuming from {len(completed_indexes)} completed sentences.")
else:
    df_checkpoint = pd.DataFrame(columns=['Truncated_Sentence', 'CTRL_xj', 'CTRL_complete'])
    completed_indexes = []

batch_size = 1
completion_list = []
combined_list = []

total_batches = len(range(0, len(df), batch_size))
for batch_num, i in enumerate(range(0, len(df), batch_size)):
    if i in completed_indexes:
        continue
    
    batch = df['Truncated_Sentence'][i:i + batch_size].tolist()

    batch_completions = generate_batch_completion_ctrl(batch, model, tokenizer, min_word_count=10)
    
    combined = [f"{sentence} {completion}" for sentence, completion in zip(batch, batch_completions)]
    
    df_temp = pd.DataFrame({'Truncated_Sentence': batch,
                            'CTRL_xj': batch_completions,
                            'CTRL_complete': combined})
    
    df_checkpoint = pd.concat([df_checkpoint, df_temp])
    df_checkpoint.to_excel(temp_save_path, index=False)
    print(f"Batch {batch_num + 1}/{total_batches} completed.")

df_checkpoint.to_excel(output_file_path, index=False)
print(f"Excel file updated and saved to: {output_file_path}")

if os.path.exists(temp_save_path):
    os.remove(temp_save_path)
    print(f"Temporary checkpoint file {temp_save_path} has been deleted.")