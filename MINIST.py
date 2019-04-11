# -*- coding: utf-8 -*-
# @Time : 2019-04-10 16:41
# @Author : Tom Chen
# @Email : chenbaocun@emails.bjut.edu.cn
# @File : MINIST.py
# @Software: PyCharm
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
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

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














#  使用二次代价函数进行梯度下降
# 第0epochAccurancy:0.7348809
# 第1epochAccurancy:0.81714284
# 第2epochAccurancy:0.8485714
# 第3epochAccurancy:0.8615476
# 第4epochAccurancy:0.86904764
# 第5epochAccurancy:0.87416667
# 第6epochAccurancy:0.8772619
# 第7epochAccurancy:0.87916666
# 第8epochAccurancy:0.88214284
# 第9epochAccurancy:0.885
# 第10epochAccurancy:0.8869048
# 第11epochAccurancy:0.88785714
# 第12epochAccurancy:0.88988096
# 第13epochAccurancy:0.89107144
# 第14epochAccurancy:0.89214283
# 第15epochAccurancy:0.8925
# 第16epochAccurancy:0.8934524
# 第17epochAccurancy:0.8945238
# 第18epochAccurancy:0.89547616
# 第19epochAccurancy:0.89619046
# 第20epochAccurancy:0.8972619
# 第21epochAccurancy:0.8972619


# 使用交叉熵代价函数做梯度下降，训练速度更快，对于softmax做最后激活函数的时候，用交叉熵更好

# 第0epochAccurancy:0.80011904
# 第1epochAccurancy:0.81714284
# 第2epochAccurancy:0.8219048
# 第3epochAccurancy:0.825119
# 第4epochAccurancy:0.85464287
# 第5epochAccurancy:0.87904763
# 第6epochAccurancy:0.8886905
# 第7epochAccurancy:0.8920238
# 第8epochAccurancy:0.89428574
# 第9epochAccurancy:0.8964286
# 第10epochAccurancy:0.8977381
# 第11epochAccurancy:0.89916664
# 第12epochAccurancy:0.90059525
# 第13epochAccurancy:0.90178573
# 第14epochAccurancy:0.9025
# 第15epochAccurancy:0.9033333
# 第16epochAccurancy:0.9039286
# 第17epochAccurancy:0.90440476
# 第18epochAccurancy:0.9052381
# 第19epochAccurancy:0.9052381
# 第20epochAccurancy:0.9054762
# 第21epochAccurancy:0.9057143