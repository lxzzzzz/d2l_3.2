import torch
import matplotlib.pyplot as plt
import random
from torch.utils import data
from torch import nn
from torch.optim import SGD

# 制作数据集 y = wx + b + error
true_w = torch.tensor([2, -3.4])
true_b = torch.tensor([4.2])

def data_set(w, b, nums):
    x = torch.normal(0, 1, (nums, len(true_w)))
    # 矩阵 * 向量
    y = torch.matmul(x, true_w) + true_b
    true_error = torch.normal(0, 0.01, y.shape)
    y += true_error
    return x, y.reshape((-1, 1))

features, targets = data_set(true_w, true_b, 1000)

lr = 0.03
Epoch = 3
batch_size = 10

# 使用Dataloader
# data.TensorDataset(features, targets) 与手写类继承dataset类，重写__getitem__相同
# 目的都是获取数据集中的元素
train_data = data.DataLoader(data.TensorDataset(features, targets), batch_size = batch_size, shuffle=True)

# 定义模型
net = nn.Sequential(nn.Linear(2, 1, bias=True, dtype=torch.float32))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 定义损失函数
loss_func = nn.MSELoss()

# 定义优化器
trainer = SGD(net.parameters(), lr = lr)

for epoch in range(Epoch):
    for i, j in train_data:
        trainer.zero_grad()
        loss = loss_func(net(i), j)
        loss.backward()
        trainer.step()
    l = loss_func(net(features), targets)
    print(l)