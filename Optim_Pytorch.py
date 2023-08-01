import torch

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
#Define an Adam optimizer to update the model parameters.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

'''
with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad'''

for t in range(500):
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

'''
import torch: 导入PyTorch库。

N, D_in, H, D_out = 64, 1000, 100, 10: 定义数据的维度和大小。

x = torch.randn(N, D_in): 生成一个随机输入张量x。

y = torch.randn(N, D_out): 生成一个随机输出张量y。

model = torch.nn.Sequential(...): 创建一个Sequential模型，按顺序包含线性层、ReLU激活函数和线性层。

torch.nn.Linear(D_in, H): 定义一个线性层，将输入维度D_in映射到隐藏层维度H。

torch.nn.ReLU(): 定义一个ReLU激活函数。

torch.nn.Linear(H, D_out): 定义一个线性层，将隐藏层维度H映射到输出维度D_out。

loss_fn = torch.nn.MSELoss(reduction='sum'): 定义一个均方误差损失函数，用于计算预测输出和真实输出之间的差异。

learning_rate = 1e-4: 设置学习率。

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate): 定义一个Adam优化器，用于更新模型参数。

for t in range(500):: 开始训练循环，执行500次迭代。

y_pred = model(x): 执行前向传播计算，将输入张量x传递给模型，得到预测输出张量y_pred。

loss = loss_fn(y_pred, y): 计算损失函数，使用均方误差来衡量预测输出和真实输出之间的差异。

if t % 100 == 99:: 每100次迭代打印当前迭代次数和损失值。

optimizer.zero_grad(): 清空优化器的梯度。

loss.backward(): 执行反向传播，计算梯度。

optimizer.step(): 使用优化器来更新模型参数。

这段代码使用了PyTorch的神经网络库（torch.nn）和损失函数库（torch.nn.functional），以及优化器（torch.optim）来自动更新模型参数。新加入的部分是优化器的定义和在训练循环中使用优化器的zero_grad()、backward()和step()函数来执行梯度清零、反向传播和参数更新的步骤。
'''