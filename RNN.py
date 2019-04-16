# -*- coding: utf-8 -*-
# @Time : 2019-04-16 11:21
# @Author : Tom Chen
# @Email : chenbaocun@emails.bjut.edu.cn
# @File : RNN.py
# @Software: PyCharm

# 使用RNN完成手写数字识别，准确率只有60%左右，后续再改进吧

import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
train=pd.read_csv('./MINISTDataset/train.csv')
y_data=train['label']
x_data=train.drop(labels=['label'],axis=1)
del train
x_data[x_data>1]=1
y_data=to_categorical(y_data,num_classes=10)
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.2,random_state=1)
# 参数
lr=0.001
training_iters=100000
batch_size=100
m_batch=batch_size//x_train.shape[0]
batch=0
n_inputs=28
n_steps=28#按行循环，每次input每一行的28的px
n_hidden_units=100#hidden layer 的神经元个数
n_classes=10#10分类
acc=np.array([])
# input
x=tf.placeholder(tf.float32,[None,n_steps,n_inputs],name='x_input')
y=tf.placeholder(tf.float32,[None,n_classes],name='y_input')

# weights
weights={
    'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
    'out':tf.Variable(tf.random_normal([n_hidden_units,n_classes]))
}

# biases
biases={
    'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units,])),
    'out':tf.Variable(tf.constant(0.1,shape=[n_classes]))
}

def RNN(X,Weights,biases):
    # hidden layer for input to cell
    # 把[100batch,28steps,28inputs]-->>[100*28,28inputs]
    X=tf.reshape(X,[-1,n_inputs])
    X_in=tf.matmul(X,weights['in']+biases['in'])
    # 转回成[100batch,28steps,100hidden]
    X_in=tf.reshape(X_in,[-1,n_steps,n_hidden_units])

    #cell,forget_bias=1表示不希望忘记前边的值

    lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias=1.0,state_is_tuple=True)
    _init_state=lstm_cell.zero_state(batch_size,dtype=tf.float32)
    outputs,state=dynamic_rnn(lstm_cell,X_in,initial_state=_init_state,time_major=False)#采用tf.rnn.dynamic_rnn调用不到，因为在init.py中缺少它的定义，所以直接在开始import进来

    # hidden layer for output as the final results
    # stete[1]是分线剧情的
    results=tf.matmul(state[1],weights['out']+biases['out'])
    return results

prediction=RNN(x,weights,biases)

# 损失函数与优化器
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
train_step=tf.train.AdamOptimizer(lr).minimize(loss)

# 准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accurancy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        if(batch>m_batch):
            batch=0
        batch_x=x_train[batch*batch_size:(batch+1)*batch_size]
        batch_y=y_train[batch*batch_size:(batch+1)*batch_size]
        # print(batch_x)
        # print(type(batch_x))
        batch_x=batch_x.values.reshape([batch_size,n_steps,n_inputs])#将Dataframe转成ndarry
        sess.run(train_step,feed_dict={x:batch_x,y:batch_y})
        batch=batch+1
        # print(x_test[0:batch_size])
        # print(type(x_test))
        if i%50==0:
            for epoch in range(x_test.shape[0] // 100):
                acc = sess.run(accurancy, feed_dict={x: x_test[epoch * 100:(epoch + 1) * 100].values.reshape([batch_size,n_steps,n_inputs]),
                                                               y: y_test[epoch * 100:(epoch + 1) * 100]},
                                         )  # axis=1时表示按行计算
                acc = np.append(acc,acc)
            print('第'+str(i)+'次迭代，准确率为：'+str(sess.run(tf.reduce_mean(acc))))

    test = pd.read_csv('./MINISTDataset/test.csv')
    test[test>0]=1
    result=np.array([])
    for epoch in range(test.shape[0] // 100):
        rst = tf.argmax(sess.run(prediction, feed_dict={x: test[epoch * 100:(epoch + 1) * 100].values.reshape([batch_size,n_steps,n_inputs])}),
                        1)  # axis=1时表示按行计算
        result = np.append(result, sess.run(rst))
    results = pd.Series(result.astype(int), name='Label')#如果数据数量不足总数的时候，astype会无效
    submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)  # 拼接axis=1表示列拼接
    submission.to_csv("./MINISTDataset/RNN.csv", index=False)


#
# 第0次迭代，准确率为：0.13
# 第50次迭代，准确率为：0.59
# 第100次迭代，准确率为：0.59
# 第150次迭代，准确率为：0.6
# 第200次迭代，准确率为：0.6
# 第250次迭代，准确率为：0.61
# 第300次迭代，准确率为：0.62
# 第350次迭代，准确率为：0.61
# 第400次迭代，准确率为：0.59
# 第450次迭代，准确率为：0.59
# 第500次迭代，准确率为：0.59
# 第550次迭代，准确率为：0.59
# 第600次迭代，准确率为：0.59
# 第650次迭代，准确率为：0.59
# 第700次迭代，准确率为：0.59
# 第750次迭代，准确率为：0.59
# 第800次迭代，准确率为：0.59
# 第850次迭代，准确率为：0.59
# 第900次迭代，准确率为：0.59
# 第950次迭代，准确率为：0.59