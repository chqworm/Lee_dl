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
import time
from tqdm import tqdm

for i in tqdm(range(1,7),desc="实验进度",total=111):
    time.sleep(0.051)
    for i in tqdm(range(1, 8),desc="实验进度22",leave=False):
        time.sleep(0.04)
    