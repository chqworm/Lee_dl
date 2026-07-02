import sys
# 数值、矩阵操作
import math
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

# 学习曲线绘制
from torch.utils.tensorboard import SummaryWriter

print(f"PyTorch Version: {torch.__version__}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# 1. 打印当前解释器的绝对路径
# print(f"当前解释器路径: {sys.executable}")

# 2. 打印 Python 搜索库的路径 (sys.path)
# print("\n库搜索路径 (sys.path):")
# for path in sys.path:
#     print(path)

# 3. 确认 Python 版本
print(f"\nPython 版本: {sys.version}")

# sys.exit(0)  # 退出程序，防止后续代码执行

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


def train_valid_split(data_set:NDArray[np.float64], valid_ratio:float, seed:int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    '''
    数据集拆分成训练集（training set）和 验证集（validation set）
    '''
    valid_set_size = int(valid_ratio * len(data_set)) 
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)


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
    
 #神经网络模型,可以在以下My_Model类框架下，进行不同结构的深度模型尝试
class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: 修改模型结构, 注意矩阵的维度（dimensions） 
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1) # (B, 1) -> (B)
        return x
    
#特征选择: 通过修改下面的函数，选择自己认为有用的特征
def select_feat(train_data, valid_data, test_data, select_all=True):
    '''
    特征选择
    选择较好的特征用来拟合回归模型
    '''
    y_train, y_valid = train_data[:,-1], valid_data[:,-1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:,:-1], valid_data[:,:-1], test_data

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = [0,1,2,3,4] # TODO: 选择需要的特征 ，这部分可以自己调研一些特征选择的方法并完善.

    return raw_x_train[:,feat_idx], raw_x_valid[:,feat_idx], raw_x_test[:,feat_idx], y_train, y_valid    


def trainer(train_loader, valid_loader, model, config, device):

    criterion = nn.MSELoss(reduction='mean') # 损失函数的定义 均方误差损失函数  ((y-y_pred)**2)/n,适合回归问题预测一个数值
    #对异常值敏感 (Outliers)：因为平方的存在，如果数据集中有极个别异常离谱的点（离群点），MSE 会给这些点赋予极高的权重。这会导致模型为了“照顾”这些离群点而牺牲掉大部分正常数据的准确度。
    #数据范围：确保你的模型输出和标签在数值规模上比较接近。如果标签是 0-1 之间的数，但你的模型输出是 1000，MSE 的值会巨大无比，可能导致训练瞬间崩溃（梯度爆炸）。
    #crossEntropyLoss: 适合分类问题，尤其是多分类问题。它衡量的是预测的概率分布与真实标签分布之间的差异。对于二分类问题，也可以使用 BCEWithLogitsLoss。

    # 定义优化器
    # TODO: 可以查看学习更多的优化器 https://pytorch.org/docs/stable/optim.html 
    # TODO: L2 正则( 可以使用optimizer(weight decay...) )或者 自己实现L2正则.
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9) 
    
    # tensorboard 的记录器
    writer = SummaryWriter()

    if not os.path.isdir('./models'):
        # 创建文件夹-用于存储模型
        os.mkdir('./models')

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train() # 训练模式
        loss_record = []

        # tqdm可以帮助我们显示训练的进度  
        train_pbar = tqdm(train_loader, position=0, leave=True)
        # 设置进度条的左边 ： 显示第几个Epoch了
        train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
        for x, y in train_pbar:
            optimizer.zero_grad()               # 将梯度置0.
            x, y = x.to(device), y.to(device)   # 将数据一到相应的存储位置(CPU/GPU)
            pred = model(x)             
            loss = criterion(pred, y)
            loss.backward()                     # 反向传播 计算梯度.
            optimizer.step()                    # 更新网络参数
            step += 1
            loss_record.append(loss.detach().item())
            
            # 训练完一个batch的数据，将loss 显示在进度条的右边
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)
        # 每个epoch,在tensorboard 中记录训练的损失（后面可以展示出来）
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval() # 将模型设置成 evaluation 模式.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())
            
        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        # 每个epoch,在tensorboard 中记录验证的损失（后面可以展示出来）
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path']) # 模型保存
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else: 
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return


# 超参设置`config` 包含所有训练需要的超参数（便于后续的调参），以及模型需要存储的位置   

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
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

# 导入数据集
#1. 从文件中读取数据`pd.read_csv`
#2. 数据拆分成三份 训练（training）、验证（validation）、测试（testing）
#    - `train_valid_split`：  分成训练、验证
#    - `select_feat`：拆分特征和label，并进行特征选择
#    - `COVID19Dataset`: 分别将训练、验证、测试集的特征和label组合成可以用于快速迭代训练的数据集`train_dataset, valid_dataset, test_dataset`
#这部分不用修改</b>

# 设置随机种子便于复现
same_seed(config['seed'])


# 训练集大小(train_data size) : 2699 x 118 (id + 37 states + 16 features x 5 days) 
# 测试集大小(test_data size）: 1078 x 117 (没有label (last day's positive rate))
pd.set_option('display.max_column', 200) # 设置显示数据的列数
train_df, test_df = pd.read_csv('./data/covid.train.csv'), pd.read_csv('./data/covid.test.csv')
print(type(train_df))
print(test_df)


#display(train_df.head(3)) # 显示前三行的样本
train_data, test_data = train_df.values, test_df.values
print(train_data[0])
# print(type(train_data))
print(type(train_data[0][0]))
print(type(train_data[0][40]))
del train_df, test_df # 删除数据减少内存占用
train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])

# 打印数据的大小
print(f"""train_data size: {train_data.shape} 
valid_data size: {valid_data.shape} 
test_data size: {test_data.shape}""")

# 特征选择
x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])

# 打印出特征数量.
print(f'number of features: {x_train.shape[1]}')

train_dataset, valid_dataset, test_dataset = COVID19Dataset(x_train, y_train), \
                                            COVID19Dataset(x_valid, y_valid), \
                                            COVID19Dataset(x_test)

# 使用Pytorch中Dataloader类按照Batch将数据集加载
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

#开始训练

model = My_Model(input_dim=x_train.shape[1]).to(device) # 将模型和训练数据放在相同的存储位置(CPU/GPU)
trainer(train_loader, valid_loader, model, config, device)

#不要再在 Notebook 中使用 %tensorboard。
#改为使用 VS Code 自带的 TensorBoard 插件（安装在左侧侧边栏中），或者直接在外部终端运行命令：
#Bash
#tensorboard --logdir=./runs/ --port=6007


#测试集的预测结果保存到`pred.csv`
def save_pred(preds, file):
    ''' 将模型保存到指定位置'''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

model = My_Model(input_dim=x_train.shape[1]).to(device)
model.load_state_dict(torch.load(config['save_path']))
preds = predict(test_loader, model, device) 
save_pred(preds, 'pred.csv')         