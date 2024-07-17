import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더
import torch

# 데이터 로드
df = pd.read_csv('/home/jaehun/Documents/STUDY/Transformer/data-02-stock_daily.csv')

# 7일간의 데이터가 입력으로 들어가고 batch size는 임의로 지정
seq_length = 7
predict_length = 3
batch = 32

# 데이터를 역순으로 정렬하여 전체 데이터의 70% 학습, 30% 테스트에 사용
df = df[::-1]  
train_size = int(len(df) * 0.7)
train_set = df.iloc[0:train_size].copy()  
test_set = df.iloc[train_size - seq_length:].copy()

# # Input scale
# scaler_x = MinMaxScaler()
# scaler_x.fit(train_set.iloc[:, :-1])

# train_set.iloc[:, :-1] = scaler_x.transform(train_set.iloc[:, :-1])
# test_set.iloc[:, :-1] = scaler_x.transform(test_set.iloc[:, :-1])

# # Output scale
# scaler_y = MinMaxScaler()
# #scaler_y.fit(train_set.iloc[:, [-1]])
# scaler_y.fit(train_set.iloc[:, :-1])

# train_set.iloc[:, -1] = scaler_y.transform(train_set.iloc[:, [-1]])
# test_set.iloc[:, -1] = scaler_y.transform(test_set.iloc[:, [-1]])

# 전체 데이터 스케일링
scaler = MinMaxScaler()
scaler.fit(train_set)

train_set = scaler.transform(train_set)
test_set = scaler.transform(test_set)

# 데이터셋 생성 함수
def build_dataset(time_series, seq_length, predict_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length - predict_length + 1):
        _x = time_series[i:i + seq_length, :]
        _y = time_series[i + seq_length:i + seq_length + predict_length, :]
        dataX.append(_x)
        dataY.append(_y)
    return np.array(dataX), np.array(dataY)

trainX, trainY = build_dataset(np.array(train_set), seq_length, predict_length)
testX, testY = build_dataset(np.array(test_set), seq_length, predict_length)

# 텐서로 변환
trainX_tensor = torch.FloatTensor(trainX)
trainY_tensor = torch.FloatTensor(trainY)

testX_tensor = torch.FloatTensor(testX)
testY_tensor = torch.FloatTensor(testY)

# 텐서 형태로 데이터 정의
dataset = TensorDataset(trainX_tensor, trainY_tensor)
evalset = TensorDataset(testX_tensor, testY_tensor)

# 데이터로더는 기본적으로 2개의 인자를 입력받으며 배치크기는 통상적으로 2의 배수를 사용
dataloader = DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=True)
evalloader = DataLoader(evalset, batch_size=batch, shuffle=False, drop_last=True)