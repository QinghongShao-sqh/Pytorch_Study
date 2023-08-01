#以下代码自定义了autograd操作ReLU非线性层，并使用它实现我们的2层神经网络：

import  torch

class MyReLU(torch.autograd.Function):
    #Indicates that the following function is a static method, which can be called without creating an instance.
    @staticmethod
    #Forward propagation function that takes input and returns the output.
    def forward(ctx,input):
        #Saves the input tensor in the context for later use in backward propagation.
        ctx.save_for_backward(input)
        #Applies the ReLU operation to the input tensor, setting values less than 0 to 0, and returns the result.
        return input.clamp(min=0)

    @staticmethod
    # Backward propagation function that takes the gradient output and returns the gradient input.
    def backward(ctx,grad_output):
        #Retrieves the input tensor from the saved context.
        input, = ctx.saved_tensors
        #Clones the gradient output to modify it.
        grad_input = grad_output.clone()
        #Sets the gradient to 0 for positions where the input is less than 0.
        grad_input[input < 0] = 0
        #Returns the modified gradient input.
        return grad_input

dtype = torch.float
device = torch.device("cpu")

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    #Creates an instance of the ReLU function.
    relu = MyReLU.apply
    #Performs forward propagation. First, it performs matrix multiplication between the input tensor x and the first layer weight tensor w1, and passes the result through the ReLU function. Then, it performs matrix multiplication between the output of the ReLU function and the second layer weight tensor w2, resulting in the predicted output tensor y_pred.
    y_pred = relu(x.mm(w1)).mm(w2)
    #Calculates the loss function, which measures the squared difference between the predicted output and the true output.
    loss = (y_pred).pow(2).sum()
    if  t % 100 == 99:
        print(t, loss.item())
    #Performs backward propagation to calculate gradients.
    loss.backward()
    #Enters a context where gradients are not computed, allowing for updating the weights without gradient calculations.
    with torch.no_grad():
        #Updates the first layer weight tensor using gradient descent.
        w1 -= learning_rate* w1.grad
        #Updates the second layer weight tensor using gradient descent.
        w2 -= learning_rate * w2.requires_grad
        #Sets the gradient of the first layer weight tensor to zero.
        w1.grad.zero_()
        #Sets the gradient of the second layer weight tensor to zero
        w1.grad.zero_()


'''
class MyReLU(torch.autograd.Function): 声明一个自定义的ReLU函数类，继承自torch.autograd.Function。

@staticmethod: 表示下面的函数是静态方法，可以在没有创建实例的情况下调用。

def forward(ctx, input): 前向传播函数，接受输入并返回结果。

ctx.save_for_backward(input): 在上下文中保存输入张量，以便在反向传播时使用。

return input.clamp(min=0): 对输入张量进行ReLU操作，将小于0的值设为0，返回结果。

def backward(ctx, grad_output): 反向传播函数，接受梯度输出并返回梯度输入。

input, = ctx.saved_tensors: 从保存的上下文中提取输入张量。

grad_input = grad_output.clone(): 克隆梯度输出，以便对其进行修改。

grad_input[input < 0] = 0: 将小于0的输入位置的梯度设为0。

return grad_input: 返回修改后的梯度输入。

dtype = torch.float: 设置张量的数据类型为浮点型。

device = torch.device("cpu"): 设置计算设备为CPU。

N, D_in, H, D_out = 64, 1000, 100, 10: 定义数据的维度和大小。

x = torch.randn(N, D_in, device=device, dtype=dtype): 生成一个随机输入张量。

y = torch.randn(N, D_out, device=device, dtype=dtype): 生成一个随机输出张量。

w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True): 随机初始化第一层权重张量，并设置requires_grad=True以便计算梯度。

w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True): 随机初始化第二层权重张量，并设置requires_grad=True以便计算梯度。

learning_rate = 1e-6: 设置学习率。

for t in range(500):: 开始训练循环，执行500次迭代。

relu = MyReLU.apply: 创建一个ReLU函数的实例。

y_pred = relu(x.mm(w1)).mm(w2): 执行前向传播计算。首先，使用输入张量x和第一层权重张量w1进行矩阵乘法，并将结果传递给ReLU函数。然后，将ReLU函数的输出和第二层权重张量w2进行矩阵乘法，得到预测的输出张量y_pred。

loss = (y_pred).pow(2).sum(): 计算损失函数，使用平方差来衡量预测输出和真实输出之间的差异。

if t % 100 == 99:: 每100次迭代打印当前迭代次数和损失值。

loss.backward(): 执行反向传播，计算梯度。

with torch.no_grad():: 进入上下文，不计算梯度，以便在更新权重时不进行梯度计算。

w1 -= learning_rate * w1.grad: 使用梯度下降更新第一层权重张量。

w2 -= learning_rate * w2.requires_grad: 使用梯度下降更新第二层权重张量。

w1.grad.zero_(): 将第一层权重张量的梯度设为零。

w1.grad.zero_(): 将第二层权重张量的梯度设为零。
'''








