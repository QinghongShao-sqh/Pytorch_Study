import torch

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
#Create a Sequential model that sequentially contains a linear layer, ReLU activation function, and another linear layer.
model = torch.nn.Sequential(
    #Define a linear layer that maps the input dimension D_in to the hidden layer dimension H.
    torch.nn.Linear(D_in,H),
    #Define a ReLU activation function.
    torch.nn.ReLU(),
    #Define a linear layer that maps the hidden layer dimension H to the output dimension D_out.
    torch.nn.Linear(H,D_out),
)
#Define a mean squared error loss function to compute the difference between predicted output and true output.
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
for t in range(500):
    #Perform forward propagation by passing the input tensor x to the model to get the predicted output tensor y_pred.
    y_pred = model(x)
    #Compute the loss function using mean squared error to measure the difference between predicted output and true output.
    loss = loss_fn(y_pred,y)
    if t%100 == 99:
        print(t,loss.item)
    # Clear all gradients of the model.
    model.zero_grad()
    #Perform backward propagation to compute gradients.
    loss.backward()
    #Perform backward propagation to compute gradients.
    with torch.no_grad():
        #Iterate over each parameter of the model.
        for param in model.parameters():
            # Update the parameters using gradient descent.
            param -= learning_rate * param.grad



'''
import torch: 导入PyTorch库。

N, D_in, H, D_out = 64, 1000, 100, 10: 定义数据的维度和大小。

x = torch.randn(N, D_in): 生成一个随机输入张量x。

y = torch.randn(N, D_out): 生成一个随机输出张量y。

model = torch.nn.Sequential(...): 创建一个Sequential模型，按顺序包含线性层、ReLU激活函数和线性层。

torch.nn.Linear(D_in,H): 定义一个线性层，将输入维度D_in映射到隐藏层维度H。

torch.nn.ReLU(): 定义一个ReLU激活函数。

torch.nn.Linear(H,D_out): 定义一个线性层，将隐藏层维度H映射到输出维度D_out。

loss_fn = torch.nn.MSELoss(reduction='sum'): 定义一个均方误差损失函数，用于计算预测输出和真实输出之间的差异。

learning_rate = 1e-4: 设置学习率。

for t in range(500):: 开始训练循环，执行500次迭代。

y_pred = model(x): 执行前向传播计算，将输入张量x传递给模型，得到预测输出张量y_pred。

loss = loss_fn(y_pred,y): 计算损失函数，使用均方误差来衡量预测输出和真实输出之间的差异。

if t%100 == 99:: 每100次迭代打印当前迭代次数和损失值。

model.zero_grad(): 清空模型的所有梯度。

loss.backward(): 执行反向传播，计算梯度。

with torch.no_grad():: 进入上下文，不计算梯度，以便在更新参数时不进行梯度计算。

for param in model.parameters():: 对模型的每个参数进行迭代。

param -= learning_rate * param.grad: 使用梯度下降更新参数。
'''







