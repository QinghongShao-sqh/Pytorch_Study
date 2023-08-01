import torch

class TwoLayerNet(torch.nn.Module):
    #Defines the initialization function of the TwoLayerNet class, which takes input dimension D_in, hidden layer dimension H, and output dimension D_out as parameters.
    def __init__(self, D_in, H, D_out):
        #Calls the initialization function of the parent class (torch.nn.Module).
        super(TwoLayerNet, self).__init__()
        #Creates a linear layer within the TwoLayerNet class that maps the input dimension D_in to the hidden layer dimension H.
        self.linear1 = torch.nn.Linear(D_in, H)
        # Creates another linear layer within the TwoLayerNet class that maps the hidden layer dimension H to the output dimension D_out.
        self.linear2 = torch.nn.Linear(H, D_out)
    #Defines the forward propagation function of the TwoLayerNet class, which takes input tensor x as a parameter.
    def forward(self, x):
        #Performs forward propagation of the first linear layer and then applies the clamp function to clip its output to non-negative values
        h_relu = self.linear1(x).clamp(min=0)
        #Performs forward propagation of the second linear layer, taking the output of the first linear layer as input.
        y_pred = self.linear2(h_relu)
        #Returns the predicted output tensor y_pred.
        return y_pred

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
#Creates an instance of the TwoLayerNet class, i.e., a model object.
model = TwoLayerNet(D_in, H, D_out)
#Defines a mean squared error loss function to compute the difference between the predicted output and the true output.
criterion = torch.nn.MSELoss(reduction='sum')
#Defines a stochastic gradient descent (SGD) optimizer to update the model parameters.
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):

    y_pred = model(x)
#Computes the loss function, using mean squared error to measure the difference between the predicted output and the true output.
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

'''
class TwoLayerNet(torch.nn.Module):: 定义一个名为TwoLayerNet的自定义神经网络类，继承自torch.nn.Module。

def __init__(self, D_in, H, D_out):: 定义TwoLayerNet类的初始化函数，接收输入维度D_in、隐藏层维度H和输出维度D_out作为参数。

super(TwoLayerNet, self).__init__(): 调用父类(torch.nn.Module)的初始化函数。

self.linear1 = torch.nn.Linear(D_in, H): 在TwoLayerNet类中创建一个线性层，将输入维度D_in映射到隐藏层维度H。

self.linear2 = torch.nn.Linear(H, D_out): 在TwoLayerNet类中创建另一个线性层，将隐藏层维度H映射到输出维度D_out。

def forward(self, x):: 定义TwoLayerNet类的前向传播函数，接收输入张量x作为参数。

h_relu = self.linear1(x).clamp(min=0): 执行第一个线性层的前向传播，然后使用clamp函数将其结果裁剪为非负值。

y_pred = self.linear2(h_relu): 执行第二个线性层的前向传播，将第一个线性层的输出作为输入。

return y_pred: 返回预测输出张量y_pred。

N, D_in, H, D_out = 64, 1000, 100, 10: 定义数据的维度和大小。

x = torch.randn(N, D_in): 生成一个随机输入张量x。

y = torch.randn(N, D_out): 生成一个随机输出张量y。

model = TwoLayerNet(D_in, H, D_out): 创建一个TwoLayerNet类的实例，即一个模型对象。

criterion = torch.nn.MSELoss(reduction='sum'): 定义一个均方误差损失函数，用于计算预测输出和真实输出之间的差异。

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4): 定义一个随机梯度下降（SGD）优化器，用于更新模型参数。

for t in range(500):: 开始训练循环，执行500次迭代。

y_pred = model(x): 执行模型的前向传播计算，将输入张量x传递给模型，得到预测输出张量y_pred。

loss = criterion(y_pred, y): 计算损失函数，使用均方误差来衡量预测输出和真实输出之间的差异。

if t % 100 == 99:: 每100次迭代打印当前迭代次数和损失值。

optimizer.zero_grad(): 清空优化器的梯度。

loss.backward(): 执行反向传播，计算梯度。

optimizer.step(): 使用优化器来更新模型参数。

这段代码定义了一个简单的两层神经网络模型，使用自定义的TwoLayerNet类继承了torch.nn.Module类，并通过重写forward函数实现了前向传播逻辑。训练循环中的步骤与前面的代码相似，包括前向传播、计算损失、梯度清零、反向传播和参数更新。不同之处在于使用的模型、损失函数和优化器'''
