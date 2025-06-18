# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer

# %%
unseen_df = pd.read_csv("data/dataset_curated_unseen.csv",keep_default_na=False, na_values=[''])
unseen_df

# %%
unseen_rpa = unseen_df['RPA'].values
rpa_values = unseen_rpa

# %%
threshold = 0.001
count_below_threshold = (rpa_values < threshold).sum()  # 统计小于 threshold 的样本数量
total_count = len(rpa_values)  # 总样本数量
percentage_below_threshold = count_below_threshold / total_count * 100  # 计算占比

print(f"Number of samples with RPA < threshold: {count_below_threshold}")
print(f"Total number of samples: {total_count}")
print(f"Percentage of samples with RPA < threshold: {percentage_below_threshold:.2f}%")


# %%
unseen_df['Affinity_Category'] = unseen_df['RPA'].apply(lambda x: 0 if x <= threshold else 1)
unseen_df

# %%
unseen_df['Fill status'] = 0

# %%
unseen_df.to_csv('unseen_data_binary.csv', index=False)


# %%

# %%
rpa_values = unseen_df['RPA'].values

# %%
# 绘制直方图
plt.hist(rpa_values, bins=100, edgecolor='black')  # bins 参数控制分箱的数量
plt.title('Histogram of RPA Values')
plt.xlabel('RPA Value')
plt.ylabel('Frequency')
plt.show()

# %%
lambda_ = np.load('../modal_fusion_hybrid/lambda_.npy', allow_pickle=True)
lambda_

# %%
def apply_boxcox(data, lambda_):
    if lambda_ == 0:
        return np.log(data)
    else:
        return (np.power(data, lambda_) - 1) / lambda_

# %%
unseen_df = unseen_df[unseen_df['Affinity_Category']==1]
unseen_df

# %%
unseen_df['Box-Cox RPA'] = apply_boxcox(unseen_df['RPA'], lambda_)
unseen_df

# %%
rpa_boxcox = unseen_df['Box-Cox RPA'].values

# %%
plt.hist(rpa_boxcox, bins=100, edgecolor='black')  # bins 参数控制分箱的数量
plt.title('Histogram of rpa_boxcox Values')
plt.xlabel('rpa_boxcox Value')
plt.ylabel('Frequency')
plt.show()

# %%
unseen_df.to_csv('unseen_data_reg.csv', index=False)

# %%
