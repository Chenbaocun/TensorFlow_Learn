# -*- coding: utf-8 -*-
# @Time : 2019-04-11 10:44
# @Author : Tom Chen
# @Email : chenbaocun@emails.bjut.edu.cn
# @File : Dropout.py
# @Software: PyCharm

# 构建了一个比较复杂的神经网络，观察过拟合现象，并且使用tf.nn.dropout 防止过拟合
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
keep_prob=tf.placeholder(tf.float32)
# 创建nn
# 第一hidden 2000个
W1=tf.Variable(tf.truncated_normal([784,2000],stddev=0.1))
bias1=tf.Variable(tf.zeros([1,2000])+0.1)
L1=tf.nn.tanh(tf.matmul(x,W1)+bias1)
L1_Dropout=tf.nn.dropout(L1,keep_prob)
# 第二hidden 2000个
W2=tf.Variable(tf.truncated_normal([2000,2000],stddev=0.1))
bias2=tf.Variable(tf.zeros([2000])+0.1)
L2=tf.nn.tanh(tf.matmul(L1_Dropout,W2)+bias2)
L2_Dropout=tf.nn.dropout(L2,keep_prob)
# 第三个hidden 1000个
W3=tf.Variable(tf.truncated_normal([2000,1000],stddev=0.1))
bias3=tf.Variable(tf.zeros([1000])+0.1)
L3=tf.nn.tanh(tf.matmul(L2_Dropout,W3)+bias3)
L3_Dropout=tf.nn.dropout(L3,keep_prob)
# 输出层10个
W4=tf.Variable(tf.truncated_normal([1000,10],stddev=0.1))
bias4=tf.Variable(tf.zeros([10])+0.1)
prrdiction=tf.nn.softmax(tf.matmul(L3_Dropout,W4)+bias4)

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
    for epoch in range(31):
        for batch in range(m_batch):
            batch_x=x_train[batch*batch_size:(batch+1)*batch_size]#loc[]以行标签取值，iloc以行号取值
            batch_y=y_train[batch*batch_size:(batch+1)*batch_size]
            sess.run(train_step,feed_dict={x:batch_x,y:batch_y,keep_prob:0.7})
        # print("第"+str(epoch)+"次训练")
        test_acc=sess.run(accuracy,feed_dict={x:x_test,y:y_test,keep_prob:1.0})
        train_acc=sess.run(accuracy,feed_dict={x:x_train,y:y_train,keep_prob:1.0})
        print("第"+str(epoch)+"epoch"+"test_Accurancy:"+str(test_acc))
        print("第"+str(epoch)+"epoch"+"train_Accurancy:"+str(train_acc))
# 未使用dropout结果：
# 第0epochtest_Accurancy:0.9179762
# 第0epochtrain_Accurancy:0.9346726
# 第1epochtest_Accurancy:0.93273807
# 第1epochtrain_Accurancy:0.95818454
# 第2epochtest_Accurancy:0.9388095
# 第2epochtrain_Accurancy:0.9683631
# 第3epochtest_Accurancy:0.9452381
# 第3epochtrain_Accurancy:0.97369045
# 第4epochtest_Accurancy:0.94607145
# 第4epochtrain_Accurancy:0.97723216
# 第5epochtest_Accurancy:0.947619
# 第5epochtrain_Accurancy:0.97943455
# 第6epochtest_Accurancy:0.9488095
# 第6epochtrain_Accurancy:0.9815476
# 第7epochtest_Accurancy:0.9509524
# 第7epochtrain_Accurancy:0.9830655
# 第8epochtest_Accurancy:0.95178574
# 第8epochtrain_Accurancy:0.98431545
# 第9epochtest_Accurancy:0.9522619
# 第9epochtrain_Accurancy:0.98520833
# 第10epochtest_Accurancy:0.9527381
# 第10epochtrain_Accurancy:0.98613095
# 第11epochtest_Accurancy:0.95357144
# 第11epochtrain_Accurancy:0.98672616
# 第12epochtest_Accurancy:0.95285714
# 第12epochtrain_Accurancy:0.9872024
# 第13epochtest_Accurancy:0.95309526
# 第13epochtrain_Accurancy:0.98732144
# 第14epochtest_Accurancy:0.95369047
# 第14epochtrain_Accurancy:0.9876786
# 第15epochtest_Accurancy:0.9538095
# 第15epochtrain_Accurancy:0.9881845
# 第16epochtest_Accurancy:0.9539286
# 第16epochtrain_Accurancy:0.9883631
# 第17epochtest_Accurancy:0.95416665
# 第17epochtrain_Accurancy:0.9886012
# 第18epochtest_Accurancy:0.95464283
# 第18epochtrain_Accurancy:0.98883927
# 第19epochtest_Accurancy:0.9540476
# 第19epochtrain_Accurancy:0.98910713
# 第20epochtest_Accurancy:0.9540476
# 第20epochtrain_Accurancy:0.98934525
# 第21epochtest_Accurancy:0.95488095
# 第21epochtrain_Accurancy:0.9895833
# 第22epochtest_Accurancy:0.95535713
# 第22epochtrain_Accurancy:0.9897619
# 第23epochtest_Accurancy:0.95559525
# 第23epochtrain_Accurancy:0.9900893
# 第24epochtest_Accurancy:0.95488095
# 第24epochtrain_Accurancy:0.9902679
# 第25epochtest_Accurancy:0.95535713
# 第25epochtrain_Accurancy:0.9905655
# 第26epochtest_Accurancy:0.9551191
# 第26epochtrain_Accurancy:0.9907738
# 第27epochtest_Accurancy:0.95559525
# 第27epochtrain_Accurancy:0.9909524
# 第28epochtest_Accurancy:0.95547616
# 第28epochtrain_Accurancy:0.9911012
# 第29epochtest_Accurancy:0.9551191
# 第29epochtrain_Accurancy:0.99125
# 第30epochtest_Accurancy:0.95559525
# 第30epochtrain_Accurancy:0.9913988


# 使用dropout之后的结果，训练的时间变慢了，最终在训练集上和测试集上进行测试的话，在训练集上准确度相同的时候，在测试集上的准确率，更高，说明在未使用dropout的时候，会有过拟合现象发生

# 第0epochtest_Accurancy:0.87785715
# 第0epochtrain_Accurancy:0.87964284
# 第1epochtest_Accurancy:0.89928573
# 第1epochtrain_Accurancy:0.9009226
# 第2epochtest_Accurancy:0.9097619
# 第2epochtrain_Accurancy:0.91261905
# 第3epochtest_Accurancy:0.91583335
# 第3epochtrain_Accurancy:0.9201488
# 第4epochtest_Accurancy:0.9204762
# 第4epochtrain_Accurancy:0.92452383
# 第5epochtest_Accurancy:0.9227381
# 第5epochtrain_Accurancy:0.9270536
# 第6epochtest_Accurancy:0.9242857
# 第6epochtrain_Accurancy:0.9305952
# 第7epochtest_Accurancy:0.9282143
# 第7epochtrain_Accurancy:0.9330952
# 第8epochtest_Accurancy:0.9277381
# 第8epochtrain_Accurancy:0.9346131
# 第9epochtest_Accurancy:0.93142855
# 第9epochtrain_Accurancy:0.93693453
# 第10epochtest_Accurancy:0.93416667
# 第10epochtrain_Accurancy:0.9394048
# 第11epochtest_Accurancy:0.9342857
# 第11epochtrain_Accurancy:0.94113094
# 第12epochtest_Accurancy:0.9354762
# 第12epochtrain_Accurancy:0.9433333
# 第13epochtest_Accurancy:0.93654764
# 第13epochtrain_Accurancy:0.9444345
# 第14epochtest_Accurancy:0.9372619
# 第14epochtrain_Accurancy:0.94595236
# 第15epochtest_Accurancy:0.9394048
# 第15epochtrain_Accurancy:0.94755954
# 第16epochtest_Accurancy:0.9395238
# 第16epochtrain_Accurancy:0.94767857
# 第17epochtest_Accurancy:0.94154763
# 第17epochtrain_Accurancy:0.94907737
# 第18epochtest_Accurancy:0.9407143
# 第18epochtrain_Accurancy:0.94988096
# 第19epochtest_Accurancy:0.94166666
# 第19epochtrain_Accurancy:0.95059526
# 第20epochtest_Accurancy:0.9411905
# 第20epochtrain_Accurancy:0.9512202
# 第21epochtest_Accurancy:0.9436905
# 第21epochtrain_Accurancy:0.95217264
# 第22epochtest_Accurancy:0.9433333
# 第22epochtrain_Accurancy:0.9533333
# 第23epochtest_Accurancy:0.945
# 第23epochtrain_Accurancy:0.9547321
# 第24epochtest_Accurancy:0.94488096
# 第24epochtrain_Accurancy:0.9551191
# 第25epochtest_Accurancy:0.94535714
# 第25epochtrain_Accurancy:0.95690477
# 第26epochtest_Accurancy:0.94607145
# 第26epochtrain_Accurancy:0.95666665
# 第27epochtest_Accurancy:0.9486905
# 第27epochtrain_Accurancy:0.95767856
# 第28epochtest_Accurancy:0.94857144
# 第28epochtrain_Accurancy:0.95872027
# 第29epochtest_Accurancy:0.9483333
# 第29epochtrain_Accurancy:0.9597321
# 第30epochtest_Accurancy:0.9492857
# 第30epochtrain_Accurancy:0.9603274