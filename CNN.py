# -*- coding: utf-8 -*-
# @Time : 2019-04-15 15:21
# @Author : Tom Chen
# @Email : chenbaocun@emails.bjut.edu.cn
# @File : CNN.py
# @Software: PyCharm

# 使用CNN进行手写数字识别，包括两个卷积层，两个全连接层

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

def weight_variable(shape):
    init = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init)

def bias_variable(shape):
    init = tf.constant(0.1,shape=shape)
    return tf.Variable(init)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME') #二维卷积层。strides指步长，前后两个参数必须都为1.中间两个参数表示在x和y方向上的步长都为1

# polling 池化，可以在信息提取的时候保留图片更多的信息，无需传入W，只需传入前边卷积层的输出x
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# 载入数据集，由于我是从kaagle教程开始的，所以数据集没有使用tensorflow内置的下载调用接口
train=pd.read_csv('./MINISTDataset/train.csv')
y_data=train['label']
x_data=train.drop(labels='label',axis=1)#1表示从column删除，0表示从index删除
x_data[x_data>0]=1
y_data=to_categorical(y_data,num_classes=10)

# 切分数据集
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.1,random_state=1)
batch_size=100
batch=0
m_batch=(x_train.shape[0]//100)

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784],name='x_input')
    y = tf.placeholder(tf.float32, [None, 10],name='y_input')
    keep_prob=tf.placeholder(tf.float32,name='keep_prob')

x_image=tf.reshape(x,[-1,28,28,1])#将数据转成28*28，-1代表不定义行数，后边为28*28，最后表示黑白图只有一个层

# conv1 layer

W_conv1=weight_variable([5,5,1,32])#5*5表示patch(滑动窗口的大小)，1表示输入image的厚度,32表示的是out_size
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)#output_size:14*14*64
h_pool1=max_pool_2x2(h_conv1)#output_size：14*14*32（由于步长变长引起的）

#conv2 layer

W_conv2=weight_variable([5,5,32,64])#5*5表示patch(滑动窗口的大小)，1表示输入image的厚度,64表示的是out_size(变厚)
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)#output_size:28*28*32
h_pool2=max_pool_2x2(h_conv2)#output_size：7*7*64（由于步长变长引起的）

# func1_layer 普通layer，全连接

W_fc1=weight_variable([7*7*64,1240])
b_fc1=bias_variable([1240])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])#[-1,7,7,64]---->>>>[-1,7*7*64]
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
# func2_layer 全连接输出层

W_fc2=weight_variable([1240,10])
b_fc2=bias_variable([10])
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2,name='prediction')

# 代价函数及优化器
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
train_step=tf.train.AdamOptimizer(1e-4).minimize(loss)

# 准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accurancy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
# 启动会话并运行定义好的op

# 实例化保存网络的类
saver=tf.train.Saver()

# 开启会话，执行定义好的op
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(5000):
        if(batch>=m_batch):
            batch=0
        batch_x = x_train[batch * batch_size:(batch + 1) * batch_size]  # loc[]以行标签取值，iloc以行号取值
        batch_y = y_train[batch * batch_size:(batch + 1) * batch_size]
        batch=batch+1
        sess.run(train_step, feed_dict={x: batch_x, y: batch_y,keep_prob:0.5})
        if epoch%50==0:
            acc = sess.run(accurancy, feed_dict={x: x_test, y: y_test, keep_prob: 0.5})
            print("第" + str(epoch) + "epoch" + "Accurancy:" + str(acc))
    saver.save(sess,'./Net/CNN/CnnNetForMINIST.ckpt')