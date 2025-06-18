# %%
# %%
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from scipy.stats import boxcox
from datetime import datetime 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from datetime import datetime
from scipy.stats import boxcox
from scipy.special import inv_boxcox

# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置随机数种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                     
    torch.cuda.manual_seed(seed)                

set_seed(42)

# %%
# 读取第一个文件
with open("protein_embedding/protein_embeddings_all.pkl", "rb") as f:
    pro_embed_dict = pickle.load(f)
    
# %%
# 检查第一个条目的数组维度
first_key = next(iter(pro_embed_dict))  # 获取第一个键
array_shape = pro_embed_dict[first_key].shape
print(f"Array shape for {first_key}: {array_shape}")


# %%
x_embed_data_non_fill = np.load("X_embedding/x_embeddings_non_fill.npy")  # 替换为你的文件路径
x_embed_data_non_fill.shape

# %%
unseen_df_all = pd.read_csv("unseen_data_reg.csv",keep_default_na=False, na_values=[''])
unseen_df_all

# %%
# ================================
# Step 2: 构建 Dataset
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
        rpa = row["Box-Cox RPA"]
        
        return torch.tensor(x_embed, dtype=torch.float32), \
               torch.tensor(pro_embed, dtype=torch.float32), \
               torch.tensor(rpa, dtype=torch.float32)

# %%
class CrossAttentionRegressor(nn.Module):
    def __init__(self, x_dim, pro_dim, hidden_dim=1024, dropout=0.3):
        super().__init__()
        # 特征投影层加入归一化和dropout 
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
        
        # 改进的注意力机制 
        self.attn  = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True,
            dropout=dropout/2  # 注意力机制内dropout 
        )
        
        # 分类器增加深度和正则化 
        self.regressor  = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim//4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//4, hidden_dim//16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//16, 1),
            nn.Identity()  # 显式声明无激活函数 
            
        )
 
        # 初始化参数 
        self._init_weights()
 
    def _init_weights(self):
        for module in [self.x_mlp, self.pro_mlp,  self.regressor]: 
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight,  nonlinearity='relu')
                    nn.init.constant_(layer.bias,  0.1)
 
    def forward(self, x_embed, pro_embed):
        # 特征投影 
        x_feat = self.x_mlp(x_embed)        # [B, 1024]
        pro_feat = self.pro_mlp(pro_embed)   # [B, 1024]
        
        # 添加序列维度 [B, 1, 1024]
        x_feat = x_feat.unsqueeze(1) 
        pro_feat = pro_feat.unsqueeze(1) 
        
        # 交叉注意力机制 
        attn_out, _ = self.attn( 
            query=x_feat,
            key=pro_feat,
            value=pro_feat,
            need_weights=False 
        )
        
        # 残差连接 
        fused_feature = x_feat + attn_out  # [B, 1, 1024]
        fused_feature = fused_feature.squeeze(1)   # [B, 1024]
        
        # 回归输出 
        reg_output  = self.regressor(fused_feature).squeeze(-1)   # [B]
        return reg_output

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

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, pro, y in test_loader:
            x, pro, y = x.to(device), pro.to(device), y.to(device)

            outputs = model(x, pro)
            preds = outputs.detach().cpu()
            all_preds.append(preds)
            all_targets.append(y.cpu())

    # 拼接成单一张量
    all_preds = torch.cat(all_preds).numpy().flatten()
    all_targets = torch.cat(all_targets).numpy().flatten()

    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    print("=== Test Set Evaluation ===")
    print(f"R2: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")

    return {
        'R2': r2,
        'MSE': mse,
        'MAE': mae,
        'Preds': all_preds
    }
    
# %% 1. 所有数据集
unseen_df = unseen_df_all.copy()

# %%
# x_embed_data_fill 在此不需要 ，x_embed_data_non_fill填充一下函数，不修改函数了
test_dataset = EmbeddingPairDataset(unseen_df, x_embed_data_non_fill, x_embed_data_non_fill, pro_embed_dict)

batch_size = 1024
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# %%
# 获取维度
x_dim = x_embed_data_non_fill.shape[1]
pro_dim = next(iter(pro_embed_dict.values())).shape[0]

# 加载已训练好的分类模型
regressor = CrossAttentionRegressor(x_dim, pro_dim)
regressor.load_state_dict(torch.load("../modal_fusion_hybrid/best_model_reg.pt"))
regressor.eval()

# %%
metrics = test_model_on_testset(regressor, test_loader)
metrics

# %%
unseen_df_all['Preds'] = metrics['Preds'].tolist()
unseen_df_all

# %%
unseen_df_all.to_csv('unseen_result_reg.csv', index=False)

# %%
