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

from sympy.codegen.ast import none
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
    'n_epochs': 30,     # 数据遍历训练次数
    'batch_size': 256,
    'learning_rate': 1e-5,
    'early_stop': 400,    # 如果early_stop轮损失没有下降就停止训练.
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
    for epoch in range(n_epochs):
        model.train() # 训练模式
        loss_record = []

        # tqdm可以帮助我们显示训练的进度  position=4,
        train_pbar = tqdm(train_loader, leave=True)
        # 设置进度条的左边 ： 显示第几个Epoch了
        train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
        for x, y in train_pbar:
            model.optimizer.zero_grad()               # 将梯度置0.
            x, y = x.to(device), y.to(device)   # 将数据一到相应的存储位置(CPU/GPU)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()                     # 反向传播 计算梯度.
            model.optimizer.step()                    # 更新网络参数
            step += 1
            loss_record.append(loss.detach().item())
            
            # 训练完一个batch的数据，将loss 显示在进度条的右边
            train_pbar.set_postfix({'loss': loss.detach().item()})

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
        #print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            #torch.save(model.state_dict(), config['save_path']) # 模型保存
            #print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else: 
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            break

    return mean_train_loss_list, mean_valid_loss_list


def write_model_stats_to_tensorboard(model_name, model_data, base_log_dir="runs"):
    """
    model_data: 对应 model_loss_dic["myModel1"]
    """
    # 1. 提取历史数据 (list of lists)
    train_history = model_data["mean_train_loss_history"]  # [[epoch1, epoch2...], [epoch1...]]
    valid_history = model_data["mean_valid_loss_history"]

    # 2. 对齐并计算均值 (处理不同 Fold 长度不一致的问题)
    def compute_mean_curve(history_list):
        print(history_list[0])
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



# 导入数据集
#1. 从文件中读取数据`pd.read_csv`
#2. 通过KFold将数据拆分成 训练（training）、验证（validation）
#    - 对数据处理，标准化，特征选择等操作，产生对应的COVID19Dataset
#    - 构建不同模型，对模型不设置
#    - 进行训练


# 设置随机种子便于复现
same_seed(config['seed'])

# 训练集大小(train_data size) : 2699 x 118 (id + 37 states + 16 features x 5 days 90个，最后一个是标签)
# 测试集大小(test_data size）: 1078 x 117 (没有label (last day's positive rate))
train_df, test_df = pd.read_csv('./data/covid.train.csv'), pd.read_csv('./data/covid.test.csv')
train_data, test_data = train_df.values, test_df.values

kf = KFold(n_splits=5, shuffle=True, random_state=42)
total_folds = kf.get_n_splits()
run_name = datetime.now().strftime("%d_%H-%M-%S")
scaler = StandardScaler()
model_loss_dic = {}
for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
    print(type(fold), type(train_idx))
    print(fold, train_idx)

    y_train, y_valid = train_data[train_idx, -1], train_data[val_idx, -1]
    x_train = train_data[train_idx,:-1]
    x_valid = train_data[val_idx,:-1]

    # 1. 复制一份，防止覆盖原数据
    x_scaled_train_data = x_train.copy()
    x_scaled_val_data = x_valid.copy()
    # 标准化
    for col in range(38, 117):
        # 1. 在训练集上 fit 并 transform
        # 注意：这里需要 reshape，因为 sklearn 要求输入为二维 (n_samples, 1)
        col_train = x_train[:, col].reshape(-1, 1)
        col_val = x_valid[:, col].reshape(-1, 1)
        # 注意：不要传染到x_scaled_val_data
        scaler.fit(col_train)
        x_scaled_train_data[:, col] = scaler.transform(col_train).flatten()
        x_scaled_val_data[:, col] = scaler.transform(col_val).flatten()

    train_dataset_no_states = COVID19Dataset(x_train,y_train),


    train_loader = DataLoader(COVID19Dataset(x_train,y_train), batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(COVID19Dataset(x_valid, y_valid), batch_size=config['batch_size'], shuffle=True, pin_memory=True)


    log_dir = os.path.join("runs", "baseline", run_name)
    # tensorboard 的记录器
    #writer = SummaryWriter(log_dir=logdir)
    # 每个epoch,在tensorboard 中记录训练的损失（后面可以展示出来）
    #writer.add_scalar('Loss/train', mean_train_loss, step)
    # 开始训练
    origin_model = myModule.Origin_Model(input_dim=x_train.shape[1]).to(device)  # 将模型和训练数据放在相同的存储位置(CPU/GPU)
    origin_model.optimizer = torch.optim.SGD(origin_model.parameters(), lr=config['learning_rate'],momentum=0.9)
    mean_train_loss, mean_valid_loss = trainer(train_loader, val_loader, origin_model, config, device)
    origin_model.__class__
    if "myModel1" not in model_loss_dic:
        model_loss_dic["myModel1"] = { "mean_train_loss_history": [],  "mean_valid_loss_history": [], }
    print(mean_train_loss)
    print(mean_valid_loss)
    model_loss_dic["myModel1"]["mean_train_loss_history"].append(mean_train_loss)
    model_loss_dic["myModel1"]["mean_valid_loss_history"].append(mean_valid_loss)


for model_name, model_data in model_loss_dic.items():
    write_model_stats_to_tensorboard(model_name, model_data)




sys.exit(0)


# 打印数据的大小
print(f"""train_data size: {train_data.shape} 
valid_data size: {valid_data.shape} 
test_data size: {test_data.shape}""")

feat_len = len(train_data[0]) #118 (id + 37 states + 16 features x 5 days)
# 特征选择
# 没有id和states
feat_idx_no_states = list(range(38, 117))
x_train_no_states, x_valid_no_states, x_test_no_states, y_train, y_valid = select_feat(train_data, valid_data, test_data,feat_idx_no_states)
feat_idx_no_id = list(range(1, 117))
x_train_no_id, x_valid_no_id, x_test_no_id, _, _ = select_feat(train_data, valid_data, test_data,feat_idx_no_id)




train_dataset_no_states, valid_dataset_no_states, test_dataset_no_states = COVID19Dataset(x_train_no_states, y_train), \
                                            COVID19Dataset(x_valid_no_states, y_valid), \
                                            COVID19Dataset(x_test_no_states)

train_dataset_no_id, valid_dataset_no_id, test_dataset_no_id = COVID19Dataset(x_train_no_id, y_train), \
                                            COVID19Dataset(x_valid_no_id, y_valid), \
                                            COVID19Dataset(x_test_no_id)


# 使用Pytorch中Dataloader类按照Batch将数据集加载
train_no_states_loader = DataLoader(train_dataset_no_states, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_no_states_loader = DataLoader(valid_dataset_no_states, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_no_states_loader = DataLoader(test_dataset_no_states, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

train_no_id_loader = DataLoader(train_dataset_no_id, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_no_id_loader = DataLoader(valid_dataset_no_id, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_no_id_loader = DataLoader(test_dataset_no_id, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

run_name = datetime.now().strftime("%d_%H-%M-%S")
log_dir = os.path.join("runs", "experiment_name", run_name)

#开始训练
model_no_states1 = myModule.My_Model_No_States(input_dim=x_train_no_states.shape[1]).to(device) # 将模型和训练数据放在相同的存储位置(CPU/GPU)
model_no_states1.optimizer = torch.optim.SGD(model_no_states1.parameters(), lr=config['learning_rate'], momentum=0.9)
trainer(train_no_states_loader, valid_no_states_loader, model_no_states1, config, device, logdir=os.path.join("runs", "11", run_name) )

model_no_states2 = myModule.My_Model_No_States(input_dim=x_train_no_states.shape[1]).to(device) # 将模型和训练数据放在相同的存储位置(CPU/GPU)
model_no_states2.optimizer = torch.optim.SGD(model_no_states2.parameters(), lr=config['learning_rate'], momentum=0.9)
trainer(train_no_states_loader, valid_no_states_loader, model_no_states2, config, device, logdir=os.path.join("runs", "22", run_name) )

sys.exit(0)


#不要再在 Notebook 中使用 %tensorboard。
#改为使用 VS Code 自带的 TensorBoard 插件（安装在左侧侧边栏中），或者直接在外部终端运行命令：
#Bash
#tensorboard --logdir=./runs/ --port=
#测试集的预测结果保存到`pred.csv`
def save_pred(preds, file):
    ''' 将模型保存到指定位置'''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

model = My_Model_No_States(input_dim=x_train_no_states.shape[1]).to(device)
model.load_state_dict(torch.load(config['save_path']))
preds = predict(test_no_states_loader, model, device)
save_pred(preds, 'pred.csv')         