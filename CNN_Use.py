# -*- coding: utf-8 -*-
# @Time : 2019-04-15 18:04
# @Author : Tom Chen
# @Email : chenbaocun@emails.bjut.edu.cn
# @File : CNN_Use.py
# @Software: PyCharm

# 调用训练好的模型
import tensorflow as tf
import pandas as pd
import numpy as np
result=np.array([])
test=pd.read_csv('./MINISTDataset/test.csv')
test[test>0]=1
# print(test)


with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./Net/CNN/CnnNetForMINIST.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./Net/CNN'))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('input/x_input:0')#name_scope一定要写上，否则找不到
    keep_prob = graph.get_tensor_by_name("input/keep_prob:0")
    predit=graph.get_tensor_by_name('prediction:0')
    for epoch in range(test.shape[0]//100):
        rst=tf.argmax(sess.run(predit,feed_dict={x:test[epoch*100:(epoch+1)*100],keep_prob: 0.5}), 1)#axis=1时表示按行计算
        result=np.append(result,sess.run(rst))
        # print(sess.run(rst))

    results = pd.Series(result, name='Label')
    submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)  # 拼接axis=1表示列拼接
    submission.to_csv("./MINISTDataset/cnn_mnist_datagen.csv", index=False)
