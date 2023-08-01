import torch
#Sets the data type to float.
dtype = torch.float
#Sets the computation device to CPU
device = torch.device("cpu")
#Defines the dimensions and sizes of the data. N represents the batch size, H represents the dimension of the hidden layer, D_in represents the input dimension, and D_out represents the output dimension.
N, H, D_in, D_out= 64, 100, 1000, 10
#Generates a random input tensor of size N rows and D_in columns
x = torch.randn(N,D_in,device =device,dtype=dtype)
#Generates a random output tensor of size N rows and D_out columns
y = torch.randn(N,D_out,device=device,dtype=dtype)
#Randomly initializes the weights tensor of the first layer with dimensions D_in rows and H columns.
w1 = torch.randn(D_in,H,device=device,dtype=dtype)
#Randomly initializes the weights tensor of the first layer with dimensions D_in rows and H columns.
w2 = torch.randn(N,D_out,device=device,dtype=dtype)
#Sets the learning rate.
learning_rate = 1e-6
#Starts the training iteration for 500 iterations.
for t in range(500):
    #Performs matrix multiplication between the input tensor x and the weights tensor w1 to compute the output tensor h of the first layer.
    h =x.mm(w1)
    #Applies the clamp function to set all elements in tensor h that are less than 0 to 0, implementing the ReLU activation function.
    h_relu = h.clamp(min = 0)
    #Performs matrix multiplication between the output tensor h_relu of the first layer and the weights tensor w2 to compute the predicted output tensor y_pred
    y_pred = h_relu.mm(w2)
    #Computes the loss function using mean squared error to measure the difference between the predicted output and the true output.
    loss = (y_pred - y).pow(2).sum().item()
    #Prints the current iteration number and loss value every 100 iterations.
    if t % 100 == 99:
        print(t , loss)
    #Computes the gradient of the loss function with respect to the predicted output tensor. Multiplying by 2.0 is for convenience in the subsequent calculations
    grad_y_pred = 2.0* (y_pred -y)
    #Computes the gradient of the loss function with respect to the weights tensor w2 of the second layer using transpose matrix multiplication
    grad_w2 = h_relu.t().mm(grad_y_pred)
    #Computes the gradient of the loss function with respect to the output tensor h_relu of the first layer using transpose matrix multiplication.
    grad_h_relu = grad_y_pred.mm(w2.t())
    #Creates a copy of the gradient of the output tensor of the first layer
    grad_h = grad_h_relu.clone()
    #Sets the gradients in the output tensor of the first layer that are less than 0 to 0, implementing the derivative of the ReLU function
    grad_h[h<0] = 0
    #Computes the gradient of the loss function with respect to the weights tensor w1 of the first layer using transpose matrix multiplication
    grad_w1 = x.t().mm(grad_h)
    #Updates the weights tensor of the first layer using gradient descent.
    w1 -= learning_rate * grad_w1
    #Updates the weights tensor of the second layer using gradient descent
    w2 -=learning_rate *grad_w2


'''
import torch：导入PyTorch库。

dtype = torch.float：设置数据类型为浮点型。

device = torch.device("cpu")：将计算设备设置为CPU。

N, H, D_in, D_out = 64, 100, 1000, 10：定义了数据的维度和大小。N表示批处理的大小，H表示隐藏层的维度，D_in表示输入的维度，D_out表示输出的维度。

x = torch.randn(N, D_in, device=device, dtype=dtype)：生成一个N行D_in列的随机输入张量。

y = torch.randn(N, D_out, device=device, dtype=dtype)：生成一个N行D_out列的随机输出张量。

w1 = torch.randn(D_in, H, device=device, dtype=dtype)：随机初始化第一层权重张量，将其维度设置为D_in行H列。

w2 = torch.randn(H, D_out, device=device, dtype=dtype)：随机初始化第二层权重张量，将其维度设置为H行D_out列。

learning_rate = 1e-6：设置学习率。

for t in range(500):：开始进行迭代训练，共进行500次。

h = x.mm(w1)：使用矩阵乘法将输入张量x与第一层权重张量w1相乘，计算第一层的输出张量h。

h_relu = h.clamp(min=0)：使用clamp函数将张量h中小于0的元素设置为0，实现ReLU激活函数。

y_pred = h_relu.mm(w2)：使用矩阵乘法将第一层的输出张量h_relu与第二层权重张量w2相乘，计算输出层的预测结果张量y_pred。

loss = (y_pred - y).pow(2).sum().item()：计算损失函数，使用均方差来度量预测结果与真实结果之间的差异。

if t % 100 == 99:：每100个迭代打印一次当前迭代的编号和损失值。

grad_y_pred = 2.0 * (y_pred - y)：计算损失函数对预测结果张量的梯度，乘以2是为了方便后续计算。

grad_w2 = h_relu.t().mm(grad_y_pred)：计算损失函数对第二层权重张量的梯度，使用转置矩阵乘法。

grad_h_relu = grad_y_pred.mm(w2.t())：计算损失函数对第一层输出张量的梯度，使用转置矩阵乘法。

grad_h = grad_h_relu.clone()：复制第一层输出张量的梯度。

grad_h[h < 0] = 0：将第一层输出张量中小于0的梯度置为0，实现ReLU函数的导数。

grad_w1 = x.t().mm(grad_h)：计算损失函数对第一层权重张量的梯度，使用转置矩阵乘法。

w1 -= learning_rate * grad_w1：使用梯度下降法更新第一层权重张量。

w2 -= learning_rate * grad_w2：使用梯度下降法更新第二层权重张量。

在每次迭代中，计算前向传播，计算损失函数，然后进行反向传播来更新权重，以此来训练神经网络模型。
'''


















