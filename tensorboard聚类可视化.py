# -*- coding: utf-8 -*-
# @Time : 2019-04-11 15:59
# @Author : Tom Chen
# @Email : chenbaocun@emails.bjut.edu.cn
# @File : tensorboard聚类可视化.py
# @Software: PyCharm
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.contrib.tensorboard.plugins import projector
# 添加聚类过程三维动画，这部分暂时还有问题。
train=pd.read_csv('./MINISTDataset/train.csv')
DIR = "D:/pycharm/TensorFlow_Learn/"
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
embedding = tf.Variable(tf.stack(x_train[:x_train.shape[0]]), trainable=False, name='embedding')
batch_size=100#一批100张图片放入网络进行训练
# 显示训练图片
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])#-1表示不确定多少航，中间是28*28矩阵，最后一个1表示单色图，RGB的话是3
    tf.summary.image('input', image_shaped_input, 10)#为了测试，值放入10张，跟怎么运行没有关系，数量在这儿写死了

# 计算总批次

m_batch=(x_train.shape[0]//100)

# 参数汇总
def variable_summary(var):
    with tf.name_scope('summary'):
        mean=tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
        with tf.name_scope('stddev'):
            stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('max',tf.reduce_max(var))
        tf.summary.scalar('min',tf.reduce_min(var))
        tf.summary.histogram('hinstogram',var)


# 创建nn
with tf.name_scope('layer'):
    with tf.name_scope('weight'):
        W = tf.Variable(tf.zeros([784, 10]),name='w')
        variable_summary(W)
    with tf.name_scope('bias'):
        bias = tf.Variable(tf.zeros([1, 10]),name='b')
        variable_summary(bias)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b=tf.matmul(x, W) + bias
    with tf.name_scope('softmax'):
        prrdiction = tf.nn.softmax(wx_plus_b)


# 二次代价函数
# loss=tf.reduce_mean(tf.square(y-prrdiction))
#使用交叉熵代价函数进行梯度下降，计算完之后需要做均值运算
with tf.name_scope('loss'):
    loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prrdiction))
    tf.summary.scalar('loss',loss)

with tf.name_scope('train'):
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init=tf.global_variables_initializer()

# accuracy
with tf.name_scope('accurancy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prrdiction,1))
    with tf.name_scope('accuracy'):
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy', accuracy)

# 创建metadata
# 合并所有的监测参数
merged=tf.summary.merge_all()

# 会话
with tf.Session() as sess:
    sess.run(init)
    if tf.gfile.Exists(DIR + 'projector/projector/metadata.tsv'):
        tf.gfile.Remove(DIR + 'projector/projector/metadata.tsv')
    with open(DIR + 'projector/projector/metadata.tsv', 'w') as f:
        labels = sess.run(tf.argmax(y_test[:], 1))#返回one-hot的1（最大）的位置
        for i in range(x_test.shape[0]):
            f.write(str(labels[i]) + '\n')
    writer=tf.summary.FileWriter('logs',sess.graph)#在当前目录的logs文件夹下存储graph

    projector_writer = tf.summary.FileWriter(DIR + 'projector/projector', sess.graph)
    saver = tf.train.Saver()
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = embedding.name
    embed.metadata_path = DIR + 'projector/projector/metadata.tsv'
    embed.sprite.image_path = DIR + 'projector/data/mnist_10k_sprite.png'
    embed.sprite.single_image_dim.extend([28, 28])
    projector.visualize_embeddings(projector_writer, config)



    for epoch in range(50):
        for batch in range(m_batch):
            batch_x=x_train[batch*batch_size:(batch+1)*batch_size]#loc[]以行标签取值，iloc以行号取值
            batch_y=y_train[batch*batch_size:(batch+1)*batch_size]
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary,_=sess.run([merged,train_step],feed_dict={x:batch_x,y:batch_y},options=run_options,run_metadata=run_metadata)#在训练的过程中进行参数的统计
            projector_writer.add_run_metadata(run_metadata, tag='step%03d' % (epoch*batch))
            projector_writer.add_summary(summary,epoch)
        # print("第"+str(epoch)+"次训练")
        acc=sess.run(accuracy,feed_dict={x:x_test,y:y_test})
        print("第"+str(epoch)+"epoch"+"Accurancy:"+str(acc))

        saver.save(sess, DIR + 'projector//a_model.ckpt', global_step=100)
        projector_writer.close()