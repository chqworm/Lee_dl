import sys
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from torch.utils.data import Dataset, DataLoader, random_split
from numpy.typing import NDArray
from sklearn.model_selection import KFold

# 假设你的数据是 X (特征) 和 y (标签)

#
# all_fold_scores = []
#
# for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
#     # 1. 切分数据
#     X_train, X_val = X[train_idx], X[val_idx]
#     y_train, y_val = y[train_idx], y[val_idx]
#
#     # 2. 必须在循环内进行预处理，防止泄露
#     # scaler = StandardScaler().fit(X_train)
#     # X_train = scaler.transform(X_train)
#     # X_val = scaler.transform(X_val)
#
#     # 3. 实例化模型 (确保每一折都是全新的模型)
#     model = MyNN()
#
#     # 4. 训练逻辑...
#     # ...
#
#     # 5. 记录这一折的评估结果
#     # all_fold_scores.append(final_score)
#
# print(f"5折交叉验证平均分: {np.mean(all_fold_scores)}")


# 假设你的数据是 numpy 数组
# 前 2 列需要缩放，后面是其他列
# data = np.random.rand(100, 5)
# # 定义你需要处理的列的索引
# column_indices = [0, 1]
# ct = ColumnTransformer(
#     [("scaler", StandardScaler(), column_indices)],
#     remainder='passthrough'
# )
# scaled_data = ct.fit_transform(data)

data = np.random.rand(100, 5)
idx = [0, 1] # 要缩放的列

# 1. 复制一份，防止覆盖原数据
scaled_data = data.copy()

train_df = pd.read_csv('./data/covid.train.csv')
train_data = train_df.values

# 2. 仅对特定列进行 fit 和 transform
scaler = StandardScaler()
scaled_data[:, idx] = scaler.fit_transform(data[:, idx])
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scaler = StandardScaler()
for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
    print(type(fold), type(train_idx))
    print(fold, train_idx)

    sub_train_data = train_data[train_idx]
    sub_val_data = train_data[val_idx]

    # 1. 复制一份，防止覆盖原数据
    scaled_sub_train_data = sub_train_data.copy()
    scaled_sub_val_data = sub_val_data.copy()
    # 标准化
    for col in range(38, 117):
        scaled_sub_train_data[:, idx] = scaler.fit_transform(sub_train_data[:, idx])
        scaled_sub_val_data[:, idx] = scaler.fit_transform(sub_val_data[:, idx])





#
#
# desc = train_df.describe()
# names = []
# for col in range(38,117):
#     col_desc = desc.iloc[:, col]
#     names.append(col_desc.name)
# # 定义处理器：不对id和州操作
# ct = ColumnTransformer(
#     [("scaler", StandardScaler(), names)],
#     remainder='passthrough' #未处理的列保持原样
# )
# # 转换数据
# scaled_data = ct.fit_transform(train_df)
#
# # 获取处理后的列名列表
# # 注意：转换后的列名顺序 = names 列表的顺序 + 剩余列的顺序
# all_columns = names + [c for c in train_df.columns if c not in names]
#
# # 重建 DataFrame
# scaled_df = pd.DataFrame(scaled_data, columns=all_columns)
# # 强制恢复成原始数据的列顺序：
# scaled_df = scaled_df[train_df.columns]
# #scaled_df.to_csv('./data/covid.scaled_train.csv', index=False)  #index=false，不把序号写入csv



# #特征选择: 通过修改下面的函数，选择自己认为有用的特征
# def select_feat(train_data:NDArray[np.float64],
#                 valid_data:NDArray[np.float64],
#                 test_data:NDArray[np.float64], feat_idx=None):
#     '''
#     特征选择
#     选择较好的特征用来拟合回归模型
#     '''
#     #最后一列
#     y_train, y_valid = train_data[:,-1], valid_data[:,-1]
#     raw_x_train, raw_x_valid, raw_x_test = train_data[:,:-1], valid_data[:,:-1], test_data
#     if feat_idx is None:
#         return raw_x_train, raw_x_valid, raw_x_test, y_train, y_valid
#     else:
#         return raw_x_train[:,feat_idx], raw_x_valid[:,feat_idx], raw_x_test[:,feat_idx], y_train, y_valid


# def train_valid_split(data_set:NDArray[np.float64], valid_ratio:float, seed:int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
#     '''
#     数据集拆分成训练集（training set）和 验证集（validation set）
#     '''
#     valid_set_size = int(valid_ratio * len(data_set))
#     train_set_size = len(data_set) - valid_set_size
#     train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
#     return np.array(train_set), np.array(valid_set)
