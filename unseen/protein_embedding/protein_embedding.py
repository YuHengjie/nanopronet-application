# %%
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import pickle
import pandas as pd
# 使用 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%
df = pd.read_excel("protein_seq_2504_unseen.xlsx", )
df

# %% 加载 tokenizer 和模型
model_path = "/yuhengjie/backup/pretrainedmodel/esm2_t36_3B_UR50D"

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16)
model = model.to(device)
model.eval()

# %%
# 设定 batch size
batch_size = 8  # 根据 GPU 显存调节

# 保存结果的 dict
embedding_dict = {}

# 遍历批次
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

    # 保存进 dict
    for acc, emb in zip(accessions, embeddings.cpu()):
        # 将 bfloat16 转换为 float32
        emb_float32 = emb.to(torch.float32)
        # 然后转换为 numpy 数组
        embedding_dict[acc] = emb_float32.numpy()
        
    # 清理显存
    del inputs
    del outputs
    del last_hidden
    del embeddings
    torch.cuda.empty_cache()  # 释放显存

# %%
# 保存到文件
with open("protein_embeddings_all.pkl", "wb") as f:
    pickle.dump(embedding_dict, f)
    
# %%
