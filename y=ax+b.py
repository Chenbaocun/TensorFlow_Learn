# -*- coding: utf-8 -*-
# @Time : 2019-04-15 10:01
# @Author : Tom Chen
# @Email : chenbaocun@emails.bjut.edu.cn
# @File : y=ax+b.py
# @Software: PyCharm

# 利用tensorflow拟合一条直线

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.2+0.5

Weight=tf.Variable(tf.random.truncated_normal([1],stddev=0.1))
biases=tf.Variable(tf.zeros([1]))
y_prediction=(x_data*Weight+biases)#这不能使用tf.softmax或者其他的激活函数，cause，y的值域和函数的值域并不相同
loss=tf.reduce_mean(tf.square(y_data-y_prediction))

train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init=tf.global_variables_initializer()
with tf.Session() as sess :
    sess.run(init)
    for step in range(1000):
        sess.run(train_step)
        if step%10 ==0:
            print(step,sess.run(Weight),sess.run(biases))
    plt.figure()
    plt.scatter(x_data, y_data,lw=3)
    plt.plot(x_data, sess.run(y_prediction), 'r-', lw=1)
    plt.show()


