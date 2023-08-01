import  torch

dtype = torch.float
device = torch.device("cpu")

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6

for t in range(500):
    #Perform the forward pass calculation. First, multiply the input tensor x with the weight tensor w1 using matrix multiplication. Then, use the clamp function to set any values less than 0 to 0. Finally, perform matrix multiplication with the weight tensor w2 to obtain the predicted output tensor y_pred.
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    #Calculate the loss function, using mean square error to measure the difference between the predicted output and the true output.
    loss = (y_pred - y).pow(2).sum()
    #Print the current iteration number and the loss value every 100 iterations
    if t % 100 == 99:
        print(t, loss.item())
    #Enter the context of no gradient calculation to update the weights without calculating gradients.
    with torch.no_grad():
        #Update the weight tensor for the first layer using gradient descent
        w1 -= learning_rate * w1.grad
        #Update the weight tensor for the second layer using gradient descent.
        w2 -= learning_rate * w2.grad
        #Reset the gradient tensor for the weights of the first layer to zero.
        w1.grad.zero_()
        #Reset the gradient tensor for the weights of the second layer to zero.
        w2.grad.zero_()


'''
import torch：导入PyTorch库。

dtype = torch.float：将数据类型设置为float。

device = torch.device("cpu")：将计算设备设置为CPU。

N, D_in, H, D_out = 64, 1000, 100, 10：定义数据的维度和大小。N表示批量大小，D_in表示输入维度，H表示隐藏层维度，D_out表示输出维度。

x = torch.randn(N, D_in, device=device, dtype=dtype)：生成一个大小为N行D_in列的随机输入张量。

y = torch.randn(N, D_out, device=device, dtype=dtype)：生成一个大小为N行D_out列的随机输出张量。

w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)：随机初始化第一层的权重张量，大小为D_in行H列，并设置requires_grad=True以便计算梯度。

w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)：随机初始化第二层的权重张量，大小为H行D_out列，并设置requires_grad=True以便计算梯度。

learning_rate = 1e-6：设置学习率。

for t in range(500):：开始进行500次训练迭代。

y_pred = x.mm(w1).clamp(min=0).mm(w2)：进行前向传播计算。首先，通过矩阵乘法将输入张量x与第一层的权重张量w1相乘，然后使用clamp函数将结果中小于0的元素设置为0，最后再与第二层的权重张量w2进行矩阵乘法运算，得到预测的输出张量y_pred。

loss = (y_pred - y).pow(2).sum()：计算损失函数，使用均方差衡量预测输出与真实输出之间的差异。

if t % 100 == 99:：每100次迭代打印当前迭代次数和损失值。

with torch.no_grad():：进入无梯度计算的上下文，以便更新权重时不计算梯度。

w1 -= learning_rate * w1.grad：使用梯度下降法更新第一层的权重张量。

w2 -= learning_rate * w2.grad：使用梯度下降法更新第二层的权重张量。

w1.grad.zero_()：将第一层的权重梯度张量重置为零。

w2.grad.zero_()：将第二层的权重梯度张量重置为零。

这段代码实现了一个简单的两层全连接神经网络的训练过程，包括前向传播、计算损失、反向传播更新权重。
'''


































