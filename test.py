import sys
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer

train_df, test_df = pd.read_csv('./data/covid.train.csv'), pd.read_csv('./data/covid.test.csv')
desc = train_df.describe()
x = desc['large_event.4']
col_0 = desc.iloc[:, 0]

print(desc.loc[['mean', 'std' ]])
data = train_df.values


# 获取第 0 列的名称
col_name = df.columns[0]

# 用名称取列
col = df[col_name]

# 假设 df 有三列：['age', 'salary', 'category']
# 我们只想对 'age' 和 'salary' 进行标准化
df = pd.DataFrame({
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000],
    'category': [1, 2, 1]
})

# 定义处理器：只对前两列操作
ct = ColumnTransformer(
    [("scaler", StandardScaler(), ['age', 'salary'])], 
    remainder='passthrough' #未处理的列保持原样
)

# 转换数据
scaled_data = ct.fit_transform(df)

# 注意：scaled_data 默认返回 numpy 数组
print(scaled_data)


#scaler = MinMaxScaler()
scaler = StandardScaler()


x_scaled = scaler.fit_transform( col )

# 实验 1：存储在 logs/exp_A
writer_a = SummaryWriter('logs/exp_A')
# 实验 2：存储在 logs/exp_B
writer_b = SummaryWriter('logs/exp_B')
# tensorboard 的记录器
writer_a.add_scalar('a', 0.1, 1)
writer_a.add_scalar('a', 0.2, 2)
writer_a.add_scalar('a', 0.3, 3)
writer_a.add_scalar('a', 0.4, 4)
writer_a.add_scalar('a', 0.5, 5)
writer_a.add_scalar('a', 0.6, 6)
writer_a.add_scalar('a', 0.7, 7)
writer_b.add_scalar('a', 0.1, 1)
writer_b.add_scalar('a', 1.2, 2)
writer_b.add_scalar('a', 11.3, 3)
writer_b.add_scalar('a', 20.4, 4)
writer_b.add_scalar('a', 30.5, 5)
writer_b.add_scalar('a', 0.6, 6)
writer_b.add_scalar('a', 0.7, 7)

writer_a.close()
writer_b.close()