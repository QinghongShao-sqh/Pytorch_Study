import numpy as np


# Define the dimensions and sizes of the data. N represents the batch size, H represents the dimension of the hidden layer, D_in represents the input dimension, and D_out represents the output dimension
N, H, D_in, D_out= 64, 100, 1000, 10
#Generate random input data with N rows and D_in columns.
x = np.random.randn(N,D_in)
#Generate random input data with N rows and D_out columns.
y = np.random.randn(N,D_out)
#Randomly initialize the weights of the first layer, setting its dimensions to D_in rows and H columns.
w1 = np.random.randn(D_in,H)
#Randomly initialize the weights of the second layer, setting its dimensions to H rows and D_out columns.
w2 = np.random.randn(H,D_out)
# Set the learning rate
learning_rate =1e-6
# Start the iterative training process for 500 iterations.
for t in range(500):
    # Compute the output of the first layer by multiplying the input data x with the weights w1 using matrix multiplication.
    # 计算第一层的输出，使用矩阵乘法将输入数据x与第一层权重w1相乘
    h =x.dot(w1)
    # Apply the ReLU activation function, setting negative values to zero.
    h_relu = np.maximum(h,0)
    # Compute the predictions of the output layer by multiplying the output of the first layer, h_relu, with the weights of the second layer, w2.
    y_pred = h_relu.dot(w2)

    #Calculate the loss function using the sum of squared differences between the predictions and the true values.
    loss = np.square(y_pred - y).sum()
    print(t,loss)
    # Compute the gradients of the loss function with respect to the predictions, multiplied by 2 for convenience.
    grad_y_pred = 2.0 * (y_pred - y)
    # Compute the gradients of the loss function with respect to the weights of the second layer using transposed matrix multiplication
    grad_w2 = h_relu.T.dot(grad_y_pred)
    # Compute the gradients of the loss function with respect to the output of the first layer using transposed matrix multiplication.
    grad_h_relu = grad_y_pred.dot(w2.T)
    # Make a copy of the gradients of the output of this first layer
    grad_h = grad_h_relu.copy()
    #Set the gradients of the output of the first layer to zero where the output is less than zero, effectively applying the derivative of the ReLU function
    grad_h[h < 0] = 0
    #Compute the gradients of the loss function with respect to the weights of the first layer using transposed matrix multiplication.
    grad_w1 = x.T.dot(grad_h)
    #Update the weights of the first layer using gradient descent
    w1 -= learning_rate * grad_w1
    #Update the weights of the second layer using gradient descent.
    w2 -= learning_rate * grad_w2

''' Chinese explaination
import numpy as np：导入NumPy库，用于进行数值计算。

N, H, D_in, D_out = 64, 100, 1000, 10：定义了数据的维度和大小。N表示批处理的大小，H表示隐藏层的维度，D_in表示输入的维度，D_out表示输出的维度。

x = np.random.randn(N, D_in)：生成一个N行D_in列的随机输入数据。

y = np.random.randn(N, D_out)：生成一个N行D_out列的随机输出数据。

w1 = np.random.randn(D_in, H)：随机初始化第一层权重，将其维度设置为D_in行H列。

w2 = np.random.randn(H, D_out)：随机初始化第二层权重，将其维度设置为H行D_out列。

learning_rate = 1e-6：设置学习率。

for t in range(500):：开始进行迭代训练，共进行500次。

h = x.dot(w1)：计算第一层的输出，使用矩阵乘法将输入数据x与第一层权重w1相乘。

h_relu = np.maximum(h, 0)：激活函数ReLU，将负值变为0。

y_pred = h_relu.dot(w2)：计算输出层的预测结果，将第一层的输出h_relu与第二层的权重w2相乘。

loss = np.square(y_pred - y).sum()：计算损失函数，使用均方差来度量预测结果与真实结果之间的差异。

grad_y_pred = 2.0 * (y_pred - y)：计算损失函数对预测结果的梯度，乘以2是为了方便后续计算。

grad_w2 = h_relu.T.dot(grad_y_pred)：计算损失函数对第二层权重的梯度，使用转置矩阵乘法。

grad_h_relu = grad_y_pred.dot(w2.T)：计算损失函数对第一层输出的梯度，使用转置矩阵乘法。

grad_h = grad_h_relu.copy()：将第一层输出的梯度复制一份。

grad_h[h < 0] = 0：将第一层输出小于0的梯度置为0，相当于ReLU的导数。

grad_w1 = x.T.dot(grad_h)：计算损失函数对第一层权重的梯度，使用转置矩阵乘法。

w1 -= learning_rate * grad_w1：更新第一层权重，使用梯度下降法进行更新。

w2 -= learning_rate * grad_w2：更新第二层权重，同样使用梯度下降法进行更新。
'''




