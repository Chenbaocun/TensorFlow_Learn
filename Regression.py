# -*- coding: utf-8 -*-
# @Time : 2019-04-10 15:34
# @Author : Tom Chen
# @Email : chenbaocun@emails.bjut.edu.cn
# @File : Regression.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]#均匀分布,并组合成200行1列
# print(x_data)
noise=np.random.normal(0,0.02,x_data.shape)#添加一些噪音
y_data=np.square(x_data)+noise

# 定义两个placeholder
x=tf.placeholder(tf.float32,[None,1])#不确定行数，但是有一列
y=tf.placeholder(tf.float32,[None,1])

# 最简单的全连接神经网络

Weight_L1=tf.Variable(tf.random_normal([1,10]))
biases_L1=tf.Variable(tf.zeros([1,10]))
Wt_plus_b_L1=tf.matmul(x,Weight_L1)+biases_L1
L1=tf.nn.tanh(Wt_plus_b_L1)

# 定义输出层

Weight_L2=tf.Variable(tf.random_normal([10,1]))
biases_L2=tf.Variable(tf.zeros([1,1]))
Wt_plus_b_L2=tf.matmul(L1,Weight_L2)+biases_L2
prediction=tf.nn.tanh(Wt_plus_b_L2)

# 二次代价函数

loss=tf.reduce_mean(tf.square(prediction-y_data))
# 梯度下降
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})

    # 获得预测值
    prediction_data=sess.run(prediction,feed_dict={x:x_data})
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_data,'r-',lw=5)
    plt.show()