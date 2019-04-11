# -*- coding: utf-8 -*-
# @Time : 2019-04-11 14:18
# @Author : Tom Chen
# @Email : chenbaocun@emails.bjut.edu.cn
# @File : tensorboard网络结构.py
# @Software: PyCharm

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
# 定义命名空间
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784],name='x_input')
    y = tf.placeholder(tf.float32, [None, 10],name='y_input')

y_train=train['label']
x_train=train.drop(labels=['label'],axis=1)
del train
x_train[x_train>0]=1
y_train=to_categorical(y_train,num_classes=10)
x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.2,random_state=0)
# print(y_train[0:])
y_test1=y_test
batch_size=100#一批100张图片放入网络进行训练
# 计算总批次

m_batch=(x_train.shape[0]//100)



# 创建nn
with tf.name_scope('layer'):
    with tf.name_scope('weight'):
        W = tf.Variable(tf.zeros([784, 10]),name='w')
    with tf.name_scope('bias'):
        bias = tf.Variable(tf.zeros([1, 10]),name='b')
    with tf.name_scope('wx_plus_b'):
        wx_plus_b=tf.matmul(x, W) + bias
    with tf.name_scope('softmax'):
        prrdiction = tf.nn.softmax(wx_plus_b)


# 二次代价函数
# loss=tf.reduce_mean(tf.square(y-prrdiction))
#使用交叉熵代价函数进行梯度下降，计算完之后需要做均值运算
with tf.name_scope('loss'):
    loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prrdiction))
with tf.name_scope('train'):
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init=tf.global_variables_initializer()

# accuracy
with tf.name_scope('accurancy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prrdiction,1))
    with tf.name_scope('accuracy'):
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# 会话
with tf.Session() as sess:
    sess.run(init)
    writer=tf.summary.FileWriter('logs',sess.graph)#在当前目录的logs文件夹下存储graph
    for epoch in range(1):
        for batch in range(m_batch):
            batch_x=x_train[batch*batch_size:(batch+1)*batch_size]#loc[]以行标签取值，iloc以行号取值
            batch_y=y_train[batch*batch_size:(batch+1)*batch_size]
            sess.run(train_step,feed_dict={x:batch_x,y:batch_y})
        # print("第"+str(epoch)+"次训练")
        acc=sess.run(accuracy,feed_dict={x:x_test,y:y_test})
        print("第"+str(epoch)+"epoch"+"Accurancy:"+str(acc))