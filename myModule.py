import torch
import torch.nn as nn


# 神经网络模型,可以在以下My_Model类框架下，进行不同结构的深度模型尝试
class Origin_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: 修改模型结构, 注意矩阵的维度（dimensions）
        self._optimizer = None
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)  # (B, 1) -> (B)
        return x

    # 1. 定义 Getter: 使用 @property 装饰器
    @property
    def optimizer(self):
        return self._optimizer

    # 2. 定义 Setter: 使用 @属性名.setter 装饰器
    @optimizer.setter
    def optimizer(self, value: torch.optim.Optimizer):
        if not isinstance(value, torch.optim.Optimizer):
            raise ValueError("必须是Optimizer")
        self._optimizer = value


class My_Model_Emebedding(nn.Module):
    def __init__(self, num_cities: int, num_numerical_features: int):
        super(My_Model_Emebedding, self).__init__()
        # TODO: 修改模型结构, 注意矩阵的维度（dimensions）

        # 1. Embedding 层：处理离散的城市ID
        self.embedding_dim = 2
        self.embedding = nn.Embedding(num_embeddings=num_cities, embedding_dim=self.embedding_dim)
        # 2. 全连接层：输入维度 = Embedding维度 + 数值特征维度
        # 假设最终输出维度是 1 (例如预测房价或流量)
        self.fc = nn.Sequential(
            nn.Linear(self.embedding_dim + num_numerical_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        city_indices = x[:num_cities]
        numerical_data = x[num_cities:num_cities + num_numerical_features]
        # 步骤 A：获取 Embedding 向量
        city_vec = self.embedding(city_indices)  # 输出形状: (batch_size, embedding_dim)
        # 步骤 B：将 Embedding 和数值特征拼接到一起
        # combined 形状: (batch_size, embedding_dim + num_numerical_features)
        combined = torch.cat([city_vec, numerical_data], dim=1)
        # 步骤 C：送入全连接层进行计算
        output = self.fc(combined)
        output = output.squeeze(1)  # (B, 1) -> (B)
        return output

