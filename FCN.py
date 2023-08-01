import random
import torch

'''
一个全连接ReLU网络，每次前向传播都选取一个1-4之间的随机数n，
我们将hidden layers的数量设置为n，也就是重复调用一个中间层n次，复用它的参数。'''
"""
       For the forward pass of the model, we randomly choose either 0, 1, 2, or 3
       and reuse the middle_linear Module that many times to compute hidden layer
       representations.

       Since each forward pass builds a dynamic computation graph, we can use normal
       Python control-flow operators like loops or conditional statements when
       defining the forward pass of the model.

       Here we also see that it is perfectly safe to reuse the same Module many
       times when defining a computational graph. This is a big improvement from Lua
       Torch, where each Module could be used only once.
       """

class DynamicNet(torch.nn.Module):
    #Defines the initialization function of the DynamicNet class, which takes input dimension D_in, hidden dimension H, and output dimension D_out as parameters.
    def __init__(self, D_in, H, D_out):
        #Calls the initialization function of the parent class (torch.nn.Module).
        super(DynamicNet, self).__init__()
        #creates a linear layer within the DynamicNet class that maps the input dimension D_in to the hidden dimension H.
        self.input_linear = torch.nn.Linear(D_in, H)
        #ates a linear layer within the DynamicNet class that maps the input dimension D_in to the hidden dimension H.
        self.middle_linear = torch.nn.Linear(H, H)
        #Creates another linear layer within the DynamicNet class that maps the hidden dimension H to the output dimension D_out.
        self.output_linear = torch.nn.Linear(H, D_out)
    #Creates another linear layer within the DynamicNet class that maps the hidden dimension H to the output dimension D_out.
    def forward(self, x):
        #Creates another linear layer within the DynamicNet class that maps the hidden dimension H to the output dimension D_out.
        h_relu = self.input_linear(x).clamp(min=0)
        #Performs a random number of iterations between hidden layers, where each iteration performs a forward propagation of a hidden layer.
        for _ in range(random.randint(0, 3)):
            #Performs the forward propagation of the hidden layer linear layer and then applies the clamp function to clip its result to non-negative values.
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        #Performs the forward propagation of the output linear layer, taking the output of the last hidden layer as input.
        return y_pred

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = DynamicNet(D_in, H, D_out)
#Defines a mean squared error loss function to compute the difference between the predicted output and the true output.
criterion = torch.nn.MSELoss(reduction='sum')
#Defines a stochastic gradient descent (SGD) optimizer with momentum to update the model parameters.
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
for t in range(500):
    y_pred = model(x)
    #Computes the loss function, using mean squared error to measure the difference between the predicted output and the true output
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())
    #Clears the gradients of the optimizer.
    optimizer.zero_grad()
    #Performs the backward propagation to compute the gradients.
    loss.backward()
    #Updates the model parameters using the optimizer.
    optimizer.step()

'''
class DynamicNet(torch.nn.Module):: 定义一个名为DynamicNet的自定义神经网络类，继承自torch.nn.Module。

def __init__(self, D_in, H, D_out):: 定义DynamicNet类的初始化函数，接收输入维度D_in、隐藏层维度H和输出维度D_out作为参数。

super(DynamicNet, self).__init__(): 调用父类(torch.nn.Module)的初始化函数。

self.input_linear = torch.nn.Linear(D_in, H): 在DynamicNet类中创建一个线性层，将输入维度D_in映射到隐藏层维度H。

self.middle_linear = torch.nn.Linear(H, H): 在DynamicNet类中创建另一个线性层，将隐藏层维度H映射到隐藏层维度H。

self.output_linear = torch.nn.Linear(H, D_out): 在DynamicNet类中创建另一个线性层，将隐藏层维度H映射到输出维度D_out。

def forward(self, x):: 定义DynamicNet类的前向传播函数，接收输入张量x作为参数。

h_relu = self.input_linear(x).clamp(min=0): 执行输入线性层的前向传播，然后使用clamp函数将其结果裁剪为非负值。

for _ in range(random.randint(0, 3)):: 在隐藏层之间进行随机次数的循环，每次循环执行一个隐藏层的前向传播。

h_relu = self.middle_linear(h_relu).clamp(min=0): 执行隐藏层线性层的前向传播，然后使用clamp函数将其结果裁剪为非负值。

y_pred = self.output_linear(h_relu): 执行输出线性层的前向传播，将最后一个隐藏层的输出作为输入。

return y_pred: 返回预测输出张量y_pred。

N, D_in, H, D_out = 64, 1000, 100, 10: 定义数据的维度和大小。

x = torch.randn(N, D_in): 生成一个随机输入张量x。

y = torch.randn(N, D_out): 生成一个随机输出张量y。

model = DynamicNet(D_in, H, D_out): 创建一个DynamicNet类的实例，即一个模型对象。

criterion = torch.nn.MSELoss(reduction='sum'): 定义一个均方误差损失函数，用于计算预测输出和真实输出之间的差异。

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9): 定义一个具有动量的随机梯度下降（SGD）优化器，用于更新模型参数。

for t in range(500):: 开始训练循环，执行500次迭代。

y_pred = model(x): 执行模型的前向传播计算，将输入张量x传递给模型，得到预测输出张量y_pred。

loss = criterion(y_pred, y): 计算损失函数，使用均方误差来衡量预测输出和真实输出之间的差异。

if t % 100 == 99:: 每100次迭代打印当前迭代次数和损失值。

optimizer.zero_grad(): 清空优化器的梯度。

loss.backward(): 执行反向传播，计算梯度。

optimizer.step(): 使用优化器来更新模型参数。

这段代码定义了一个动态的神经网络模型，使用自定义的DynamicNet类继承了torch.nn.Module类，并通过重写forward函数实现了前向传播逻辑。训练循环中的步骤与前面的代码相似，包括前向传播、计算损失、梯度清零、反向传播和参数更新。不同之处在于使用的模型、损失函数、优化器以及引入了随机循环的隐藏层。
'''
