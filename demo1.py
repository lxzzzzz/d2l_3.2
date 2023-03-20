import torch
import matplotlib.pyplot as plt
import random
from torch.utils import data

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
# plt.scatter(features[:, 1], targets[:, 0])
# plt.show()

def inter(features, targets, batch_size):
    indict = list(range(0, len(features)))
    random.shuffle(indict)
    for i in range(0, len(features), batch_size):
        indict_ = indict[i: i + batch_size]
        yield features[indict_], targets[indict_]

# for bf, bt in inter(features, targets, batch_size):
#     print(bf, '\n', bt)
#     break

# 定义模型
w = torch.normal(0, 0.01, (2,), requires_grad=True)
b = torch.zeros((1,), requires_grad=True)


def liner_reg(x, w, b):
    # shape(10,)
    return torch.matmul(x, w) + b

# 定义损失函数
def loss_func(y_pred, y):
    # shape(10,)
    return (y_pred - y.reshape(y_pred.shape)) ** 2 / 2


# 定义优化器
def sgd(paramters, lr, batch_size):
    with torch.no_grad():
        for i in paramters:
            i.data -= lr * i.grad.data / batch_size
            i.grad.zero_()

lr = 0.03
Epoch = 3
batch_size = 10

# 训练
for epoch in range(Epoch):
    for x, y in inter(features, targets, batch_size):
        loss = loss_func(liner_reg(x, w, b), y)
        loss.backward(torch.ones(len(loss)))
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        loss1 = loss_func(liner_reg(features, w, b), targets)
        print(loss1.mean())