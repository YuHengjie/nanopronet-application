# %%
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import random

# Setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                     
    torch.cuda.manual_seed(seed)                

set_seed(42)

# %%
# read the protein embeddings
with open("protein_embedding/protein_embeddings_all.pkl", "rb") as f:
    pro_embed_dict = pickle.load(f)
    
# %%
# check the shape of the first entry in the dictionary
first_key = next(iter(pro_embed_dict))  # 获取第一个键
array_shape = pro_embed_dict[first_key].shape
print(f"Array shape for {first_key}: {array_shape}")

# %%
x_embed_data_non_fill = np.load("X_embedding/x_embeddings_non_fill.npy")  # 替换为你的文件路径
x_embed_data_non_fill.shape

# %%
unseen_df_all = pd.read_csv("unseen_data_binary.csv",keep_default_na=False, na_values=[''])
unseen_df_all

# %%
# ================================
# create Dataset class for embedding pairs
# ================================
class EmbeddingPairDataset(Dataset):
    def __init__(self, df, x_embed_data_fill, x_embed_data_non_fill, pro_embed_dict):
        self.df = df.reset_index(drop=True)
        self.x_embed_data_fill = x_embed_data_fill
        self.x_embed_data_non_fill = x_embed_data_non_fill
        self.pro_embed_dict = pro_embed_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Check the 'Fill status' and select the appropriate embedding
        if row['Fill status'] == 1:
            x_embed = self.x_embed_data_fill[int(row["x_index"])]
        else:
            x_embed = self.x_embed_data_non_fill[int(row["x_index"])]
        
        # Fetch the protein embedding
        pro_embed = self.pro_embed_dict[row["Accession"]]
        
        # Affinity category
        rpa = row["Affinity_Category"]
        
        return torch.tensor(x_embed, dtype=torch.float32), \
               torch.tensor(pro_embed, dtype=torch.float32), \
               torch.tensor(rpa, dtype=torch.float32)
               
# %%
class CrossAttentionClassifier(nn.Module):
    def __init__(self, x_dim, pro_dim, hidden_dim=1024, dropout=0.3):
        super().__init__()
        # projecting x and pro embeddings to a common space
        self.x_mlp = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.pro_mlp  = nn.Sequential(
            nn.Linear(pro_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # cross-attention mechanism
        self.attn  = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True,
            dropout=dropout/2  # dropout 
        )
        
        # classification head
        self.classifier  = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim//4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//4, hidden_dim//16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//16, 1)  # binary classification output
        )
 
        # initialize weights
        self._init_weights()
 
    def _init_weights(self):
        for module in [self.x_mlp, self.pro_mlp,  self.classifier]: 
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight,  nonlinearity='relu')
                    nn.init.constant_(layer.bias,  0.1)
 
    def forward(self, x_embed, pro_embed):
        # project embeddings to a common space
        x_feat = self.x_mlp(x_embed)        # [B, 1024]
        pro_feat = self.pro_mlp(pro_embed)   # [B, 1024]
        
        # add batch dimension for attention mechanism
        x_feat = x_feat.unsqueeze(1) 
        pro_feat = pro_feat.unsqueeze(1) 
        
        # cross-attention
        attn_out, _ = self.attn( 
            query=x_feat,
            key=pro_feat,
            value=pro_feat,
            need_weights=False 
        )
        
        # fuse features
        fused_feature = x_feat + attn_out  # [B, 1, 1024]
        fused_feature = fused_feature.squeeze(1)   # [B, 1024]
        
        # classification output 
        logits = self.classifier(fused_feature).squeeze(-1)   # [B]
        return logits  # for binary classification, logits are used directly with BCEWithLogitsLoss
    
# %%
def test_model_on_testset(model, test_loader,):
    """
    Evaluates the trained classification model on the test dataset.

    Args:
        model (nn.Module): Trained classification model.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (str): Device to use for evaluation.

    Returns:
        dict: A dictionary containing test metrics.
    """
    model.to(device)
    model.eval()

    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, pro, y in test_loader:
            x, pro, y = x.to(device), pro.to(device), y.to(device)
            
            # Forward pass
            outputs = model(x, pro)
            # Get probabilities using sigmoid (BCEWithLogitsLoss expects logits)
            probs = torch.sigmoid(outputs).detach().cpu()
            # 确保将 loaded_threshold_binary 转换为 PyTorch tensor
            threshold_tensor = torch.tensor(loaded_threshold_binary, device=probs.device)
            preds = (probs > threshold_tensor).long().squeeze()
            
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds,) 
    rec = recall_score(all_targets, all_preds,)
    f1 = f1_score(all_targets, all_preds, )
    cm = confusion_matrix(all_targets, all_preds)
    auc_score = roc_auc_score(all_targets, all_probs)

    print("=== Test Set Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 score': f1,
        'AUC score': auc_score,
        'Confusion matrix': cm,
        'Probs': all_probs
    }

# %%
loaded_threshold_binary = np.load('../modal_fusion_hybrid/threshold_binary.npy', allow_pickle=True)
loaded_threshold_binary = min(loaded_threshold_binary, 0.2) #取较小值，recall更重要，牺牲一定的precision, 提高recall
loaded_threshold_binary

# %% 1.所有测试集
unseen_df = unseen_df_all.copy()

# Create datasets with the modified class
test_dataset = EmbeddingPairDataset(unseen_df, x_embed_data_non_fill, x_embed_data_non_fill, pro_embed_dict)

batch_size = 1024
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# %%
# 获取维度
x_dim = x_embed_data_non_fill.shape[1]
pro_dim = next(iter(pro_embed_dict.values())).shape[0]

# 加载已训练好的分类模型
classifier = CrossAttentionClassifier(x_dim, pro_dim)
classifier.load_state_dict(torch.load("../modal_fusion_hybrid/best_model_binary.pt"))
classifier.eval() 

# %% 所有测试集
print('All test dataset:')
metrics = test_model_on_testset(classifier, test_loader)
metrics

# %%
unseen_df_all['Probs'] = metrics['Probs'].tolist()
unseen_df_all

# %%
unseen_df_all.to_csv('unseen_result_binary.csv', index=False)

# %%
