from datetime import datetime
import sys
# 数值、矩阵操作
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

# 数据读取与写入make_dot
import pandas as pd
import os
import csv

# 进度条
from tqdm import tqdm
# 如果是使用notebook 推荐使用以下（颜值更高 : ) ）
# from tqdm.notebook import tqdm

# Pytorch 深度学习张量操作框架
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
# 绘制pytorch的网络
from torchviz import make_dot
import myModule

# 学习曲线绘制
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# 超参设置`config` 包含所有训练需要的超参数（便于后续的调参），以及模型需要存储的位置
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 5201314,      # 随机种子，可以自己填写. :)
    'select_all': True,   # 是否选择全部的特征
    'valid_ratio': 0.2,   # 验证集大小(validation_size) = 训练集大小(train_size) * 验证数据占比(valid_ratio)
    'n_epochs': 2000,     # 数据遍历训练次数
    'batch_size': 256,
    'learning_rate': 1e-5,
    'early_stop': 1000,    # 如果early_stop轮损失没有下降就停止训练.
    'save_path': './models/model.ckpt'  # 模型存储的位置
}


# 1. 打印当前解释器的绝对路径
# print(f"当前解释器路径: {sys.executable}")
# 2. 打印 Python 搜索库的路径 (sys.path)
# print("\n库搜索路径 (sys.path):")
# for path in sys.path:
#     print(path)
# 3. 确认 Python 版本
#print(f"\nPython 版本: {sys.version}")


# 可以不做修改
def same_seed(seed:int) -> None:
    '''
    设置随机种子(便于复现)
    '''
    #在处理卷积运算时,不要使用那些虽然快但具有不确定性Non-deterministic的优化算法。
    torch.backends.cudnn.deterministic = True
    #关闭 cuDNN 的自动调优（Auto-tuner）。通常它会根据显卡架构寻找最快的算法，
    # 但这会导致计算过程有细微的随机波动，设置为 False 可以保证一致性。
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f'Set Seed = {seed}')




def predict(test_loader, model, device):
    model.eval() # 设置成eval模式.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)                        
        with torch.no_grad():
            pred = model(x)         
            preds.append(pred.detach().cpu())   
    preds = torch.cat(preds, dim=0).numpy()  
    return preds

class COVID19Dataset(Dataset):
    '''
    x: np.ndarray  特征矩阵.
    y: np.ndarray  目标标签, 如果为None,则是预测的数据集
    '''
    def __init__(self, x:np.ndarray, y:np.ndarray=None)-> None:
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):  #tuple[torch.FloatTensor, torch.FloatTensor]|torch.FloatTensor
        if self.y is None:
            return self.x[idx]
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)
    

def trainer(train_loader:DataLoader, valid_loader:DataLoader, 
            model:nn.Module, config:dict, device:torch.device|str)-> tuple[list[Any], list[Any]] :

    criterion = nn.MSELoss(reduction='mean') # 损失函数的定义 均方误差损失函数  ((y-y_pred)**2)/n,适合回归问题预测一个数值
    #对异常值敏感 (Outliers)：因为平方的存在，如果数据集中有极个别异常离谱的点（离群点），MSE 会给这些点赋予极高的权重。这会导致模型为了“照顾”这些离群点而牺牲掉大部分正常数据的准确度。
    #数据范围：确保你的模型输出和标签在数值规模上比较接近。如果标签是 0-1 之间的数，但你的模型输出是 1000，MSE 的值会巨大无比，可能导致训练瞬间崩溃（梯度爆炸）。
    #crossEntropyLoss: 适合分类问题，尤其是多分类问题。它衡量的是预测的概率分布与真实标签分布之间的差异。对于二分类问题，也可以使用 BCEWithLogitsLoss。

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    mean_train_loss_list = []
    mean_valid_loss_list = []
    
    pbar = tqdm(range(n_epochs), desc= model.__class__.__name__ , total=n_epochs,leave=True)
    for epoch in pbar:
        model.train() # 训练模式
        loss_record = []
        
        # 训练一个批次
        for x, y in train_loader:
            model.optimizer.zero_grad()               # 将梯度置0.
            x, y = x.to(device), y.to(device)   # 将数据一到相应的存储位置(CPU/GPU)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()                     # 反向传播 计算梯度.
            model.optimizer.step()                    # 更新网络参数
            step += 1
            loss_record.append(loss.detach().item())

        mean_train_loss = sum(loss_record)/len(loss_record)
        mean_train_loss_list.append( mean_train_loss )
        


        model.eval() # 将模型设置成 evaluation 模式.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)
            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        mean_valid_loss_list.append( mean_valid_loss )
        #pbar.set_postfix({'平均训练损失': mean_train_loss, '平均验证损失': mean_valid_loss})

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            #torch.save(model.state_dict(), config['save_path']) # 模型保存
            #print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else: 
            early_stop_count += 1

        # if early_stop_count >= config['early_stop']:
        #     print('\nModel is not improving, so we halt the training session.')
        #     break

    return mean_train_loss_list, mean_valid_loss_list


def write_model_stats_to_tensorboard(model_name, model_data, base_log_dir:str="runs"):
    """
    model_data: 对应 model_loss_dic["myModel1"]
    """
    # 1. 提取历史数据 (list of lists)
    train_history = model_data["mean_train_loss_history"]  # [[epoch1, epoch2...], [epoch1...]]
    valid_history = model_data["mean_valid_loss_history"]

    # 2. 对齐并计算均值 (处理不同 Fold 长度不一致的问题)
    def compute_mean_curve(history_list):
        max_len = max(len(h) for h in history_list)
        # 用最后一个值填充，确保矩阵整齐
        aligned = np.full((len(history_list), max_len), np.nan)
        for i, h in enumerate(history_list):
            aligned[i, :len(h)] = h      # 把该 Fold 真实的 Loss 填进去
            aligned[i, len(h):] = h[-1]  # 把剩下的空间用“最后一次的值”填满
        return np.nanmean(aligned, axis=0), np.nanstd(aligned, axis=0)

    mean_train, std_train = compute_mean_curve(train_history)
    mean_valid, std_valid = compute_mean_curve(valid_history)

    # 3. 写入 TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(base_log_dir, model_name))

    for epoch in range(len(mean_train)):
        writer.add_scalar('Loss/mean_train', mean_train[epoch], epoch)
        #writer.add_scalar('Loss/std_train', std_train[epoch], epoch)
        writer.add_scalar('Loss/mean_valid', mean_valid[epoch], epoch)
        #writer.add_scalar('Loss/std_valid', std_valid[epoch], epoch)

    writer.close()
    print(f"模型 {model_name} 的统计曲线已写入 TensorBoard。")

def trainer2(x_train, y_train,x_valid,y_valid, experiment_name: str, model:nn.Module,model_loss_dic:dict, config:dict, device:torch.device|str)-> None :
    train_loader = DataLoader(COVID19Dataset(x_train, y_train), batch_size=config['batch_size'], shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(COVID19Dataset(x_valid, y_valid), batch_size=config['batch_size'], shuffle=True,
                            pin_memory=True)
    mean_train_loss, mean_valid_loss = trainer(train_loader, val_loader, model, config, device)
    
    if experiment_name not in model_loss_dic:
        model_loss_dic[experiment_name] = {"mean_train_loss_history": [], "mean_valid_loss_history": [], }
    model_loss_dic[experiment_name]["mean_train_loss_history"].append(mean_train_loss)
    model_loss_dic[experiment_name]["mean_valid_loss_history"].append(mean_valid_loss)

# 导入数据集
#1. 从文件中读取数据`pd.read_csv`
#2. 通过KFold将数据拆分成 训练（training）、验证（validation）
#    - 对数据处理，标准化，特征选择等操作，产生对应的COVID19Dataset
#    - 构建不同模型，对模型不设置
#    - 进行训练


# 设置随机种子便于复现
same_seed(config['seed'])

# 训练集大小(train_data size) : 2699 x 118 (id + 37 states + 16 features x 5 days 80个，最后一个是标签)
# 测试集大小(test_data size）: 1078 x 117 (没有label (last day's positive rate))
train_df = pd.read_csv('./data/covid.train.csv')
train_data = train_df.values
#把第一列id去掉
train_data = train_data[:,1:]

# test_df = pd.read_csv('./data/covid.test.csv')
# test_data = test_df.values
# test_data = test_data[:,1:]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
total_folds = kf.get_n_splits()
scaler = StandardScaler()
model_loss_dic = {}
# 1. 外层进度条：监控 Fold
for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):

    y_train, y_valid = train_data[train_idx, -1], train_data[val_idx, -1]
    x_train = train_data[train_idx,:-1]
    x_valid = train_data[val_idx,:-1]
    # 1. 复制一份，防止覆盖原数据
    x_scaled_train = x_train.copy()
    x_scaled_val = x_valid.copy()
    # 标准化
    for col in range(37, 116):
        # 1. 在训练集上 fit 并 transform
        # 注意：这里需要 reshape，因为 sklearn 要求输入为二维 (n_samples, 1)
        col_train = x_train[:, col].reshape(-1, 1)
        col_val = x_valid[:, col].reshape(-1, 1)
        # 注意：不要传染到x_scaled_val_data
        scaler.fit(col_train)
        x_scaled_train[:, col] = scaler.transform(col_train).flatten()
        x_scaled_val[:, col] = scaler.transform(col_val).flatten()

    models = []
    originModel = myModule.OriginModel(input_dim=x_train.shape[1]).to(device)  # 将模型和训练数据放在相同的存储位置(CPU/GPU)
    originModel.optimizer = torch.optim.SGD(originModel.parameters(), lr=config['learning_rate'],momentum=0.9)
    models.append({
    "experiment_name": "simple",
    "model": originModel,
    "x_train": x_train,
    "x_valid": x_valid,
    "description": "最简单的模型，没有标准化，没有去除独热数据"
    })

    x_train_feature = x_train[:, 37:]
    x_valid_feature = x_valid[:, 37:]

    originModel2 = myModule.OriginModel(input_dim=x_train_feature.shape[1]).to(device)  # 将模型和训练数据放在相同的存储位置(CPU/GPU)
    originModel2.optimizer = torch.optim.SGD(originModel2.parameters(), lr=config['learning_rate'],momentum=0.9)
    models.append({
    "experiment_name": "simplefeature",    
    "model": originModel2,
    "x_train": x_train_feature,
    "x_valid": x_valid_feature,
    "description": "最简单的模型，没有标准化，去除独热数据"
    })

    
    # improveModel1 = myModule.ImproveModel1(input_dim=x_train_feature.shape[1]).to(device)
    # improveModel1.optimizer = torch.optim.Adam(improveModel1.parameters(), lr=config['learning_rate'] )
    # models.append({
    # "experiment_name": "improveModel1",
    # "model": improveModel1,
    # "x_train": x_train_feature,
    # "x_valid": x_valid_feature,
    # "description": "模型层多一些，去除独热数据"
    # })

    # x_scaled_train_feature = x_scaled_train[:, 37:]
    # x_scaled_val_feature = x_scaled_val[:, 37:]
    # improveModel1 = myModule.ImproveModel1(input_dim=x_scaled_train_feature.shape[1]).to(device)
    # improveModel1.optimizer = torch.optim.Adam(improveModel1.parameters(), lr=config['learning_rate'] )
    # models.append({
    # "experiment_name": "improveModel1scaled",
    # "model": improveModel1,
    # "x_train": x_scaled_train_feature,
    # "x_valid": x_scaled_val_feature,
    # "description": "模型层多一些，去除独热数据，标准化数据"
    # })


    # improveModel2 = myModule.ImproveModel2( 37 , 79).to(device)
    # improveModel2.optimizer = torch.optim.Adam(improveModel2.parameters(), lr=config['learning_rate'])
    # models.append({
    # "model": improveModel2,
    # "x_train": x_scaled_train,
    # "x_valid": x_scaled_val,
    # "description": "模型层多一些,独热数据embedding,标准化数据"
    # })
    train_pbar = tqdm(models, leave=False, desc=f"Fold {fold+1}/{total_folds}", total=len(models))
    for model_info in train_pbar:
        train_pbar.set_postfix({"模型描述": model_info["description"]})
        trainer2(model_info["x_train"], y_train, model_info["x_valid"], 
                 y_valid, model_info["experiment_name"], model_info["model"], model_loss_dic, config, device)   
    break  # 只跑一个Fold，便于调试，后续可以去掉

    

run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f'runs/experiment_{run_name}'
for model_name, model_data in model_loss_dic.items():
    write_model_stats_to_tensorboard(model_name, model_data , base_log_dir=log_dir)



