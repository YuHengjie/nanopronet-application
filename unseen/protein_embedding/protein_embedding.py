# %%
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import pickle
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %% Example
df_all = pd.read_csv("../data/dataset_curated_unseen.csv",keep_default_na=False, na_values=[''])
df_all

# %% Accession with sequence. For new protein, use uniprot api to get sequences
df_seq = pd.read_excel('protein_seq_2504_unseen.xlsx')
df_seq

# %%
# Get unique Accession values from df_all
unique_accessions = df_all['Accession'].unique()

# Filter df_seq to keep only rows where 'Accession' is in unique_accessions
df_seq_filtered = df_seq[df_seq['Accession'].isin(unique_accessions)]
df_seq_filtered

# %%
df = df_seq_filtered.copy()

# %% load esm2 model (esm2_t36_3B_UR50D)
model_path = "/yuhengjie/backup/pretrainedmodel/esm2_t36_3B_UR50D"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16)
model = model.to(device)
model.eval()

# %%
# set batch size
batch_size = 8  # based on GPU memory capacity, adjust as needed

# create a dictionary to store embeddings
embedding_dict = {}

# loop through the dataframe in batches
for i in tqdm(range(0, len(df), batch_size), desc="Encoding"):
    batch_df = df.iloc[i:i+batch_size]
    accessions = batch_df["Accession"].tolist()
    sequences = batch_df["Sequence"].tolist()

    # tokenize
    inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state  # (B, L, H)

        # attention mask-based mean pooling
        attention_mask = inputs["attention_mask"]
        embeddings = (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)

    # save embeddings to dictionary
    for acc, emb in zip(accessions, embeddings.cpu()):
        # convert to float32 for compatibility
        emb_float32 = emb.to(torch.float32)
        # convert to numpy array
        embedding_dict[acc] = emb_float32.numpy()
        
    # clean up memory
    del inputs
    del outputs
    del last_hidden
    del embeddings
    torch.cuda.empty_cache()  # 释放显存

# %%
# save the embeddings to a pickle file
with open("protein_embeddings_all.pkl", "wb") as f:
    pickle.dump(embedding_dict, f)
    
# %%
