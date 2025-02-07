from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

import pandas as pd
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("ElnaggarLab/ankh2-ext2", trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained("ElnaggarLab/ankh2-ext2", trust_remote_code=True).to(device)

decoder_input_ids = tokenizer("<s>", return_tensors="pt").input_ids.to(device)

df_sequences = pd.read_csv(sys.argv[1])
df_sequences = df_sequences[:2]

column_seq = sys.argv[2]
column_response = sys.argv[3]
name_export = sys.argv[4]

matrix_embedding = []
response_columns = []

for index in df_sequences.index:
    sequence = df_sequences[column_seq][index]
    response_columns.append(df_sequences[column_response][index])

    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True).to(device)
    outputs = model(input_ids=inputs["input_ids"], decoder_input_ids=decoder_input_ids)

    emb = outputs.encoder_last_hidden_state
    protein_emb = emb[0].mean(dim=0)

    sequence_embedding = protein_emb.detach().cpu().numpy().squeeze()

    matrix_embedding.append(sequence_embedding)

torch.cuda.empty_cache()

header = [f"p_{i+1}" for i in range(len(matrix_embedding[0]))]

df_coded = pd.DataFrame(data=matrix_embedding, columns=header)
df_coded[column_response] = response_columns

df_coded.to_csv(name_export, index=False)
