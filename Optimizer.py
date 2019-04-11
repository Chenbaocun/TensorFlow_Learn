# -*- coding: utf-8 -*-
# @Time : 2019-04-11 13:51
# @Author : Tom Chen
# @Email : chenbaocun@emails.bjut.edu.cn
# @File : Optimizer.py
# @Software: PyCharm

# 继续在MINIST上对tf中不同的optimizer进行试验

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
train=pd.read_csv('./MINISTDataset/train.csv')
# print(minist)
y_train=train['label']
x_train=train.drop(labels=['label'],axis=1)
del train
x_train[x_train>0]=1
y_train=to_categorical(y_train,num_classes=10)
x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.2,random_state=0)#随机数种子，当种子相同时每次都会产生相同的随机数，则数据集的划分结果也是相同的。若不设置的话会随机选择一个种子
# print(y_train[0:])
y_test1=y_test
batch_size=100#一批100张图片放入网络进行训练
# 计算总批次

m_batch=(x_train.shape[0]//100)

x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])

# 创建nn

W=tf.Variable(tf.zeros([784,10]))
bias=tf.Variable(tf.zeros([1,10]))
prrdiction=tf.nn.softmax(tf.matmul(x,W)+bias)

# 二次代价函数
# loss=tf.reduce_mean(tf.square(y-prrdiction))
#使用交叉熵代价函数进行梯度下降，计算完之后需要做均值运算
loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prrdiction))
# train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)#或者1e-3
init=tf.global_variables_initializer()

# accuracy
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prrdiction,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# 会话
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(1000):
        for batch in range(m_batch):
            batch_x=x_train[batch*batch_size:(batch+1)*batch_size]#loc[]以行标签取值，iloc以行号取值
            batch_y=y_train[batch*batch_size:(batch+1)*batch_size]
            sess.run(train_step,feed_dict={x:batch_x,y:batch_y})
        # print("第"+str(epoch)+"次训练")
        acc=sess.run(accuracy,feed_dict={x:x_test,y:y_test})
        print("第"+str(epoch)+"epoch"+"Accurancy:"+str(acc))