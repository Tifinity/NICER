'''
Time: 2018/12/10
Author: Tifinity
'''
import tensorflow as tf
import numpy as np
import csv
import random

filename = './train_data.csv'
model_path = './tensorflow_net/model.ckpt'

x_data = np.array([[0, 0, 0]])
y_data = np.array([[0, 0, 0]])
with open(filename) as f:
    reader = csv.reader(f)
    st = list(reader)
    print(len(st))
    for k in range(10000):
        i = st[random.randint(0,len(st)-1)]
        x = []
        count = 0
        i[0] = i[0].strip('()').replace(' ', '')
        #print(i[0])
        s = ""
        for j in i[0]:
            if(j == ','):
                x.append(int(s))
                count += 1
                s = ""
            else:
                s += j
        y = []
        count = 0
        i[1] = i[1].strip('()').replace(' ', '')
        #print(i[1])
        s = ""
        for j in i[1]:
            if(j == ','):
                y.append(int(s))
                count += 1
                s = ""
            else:
                s += j
        x_data = np.r_[x_data, np.array([[x[0], x[1], x[2]]])]
        y_data = np.r_[y_data, np.array([[y[0], y[1], y[2]]])]

x_data = np.delete(x_data, 0, 0)
y_data = np.delete(y_data, 0, 0)

def add_layer(inputs, in_size, out_size, activation_function = None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]), dtype=tf.float32, name='w')
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, dtype=tf.float32, name='b')
    wx_plus_b = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b)
    return outputs

xs = tf.placeholder(tf.float32, [x_data.shape[0], x_data.shape[1]])
ys = tf.placeholder(tf.float32, [y_data.shape[0], y_data.shape[1]])

l1 = add_layer(xs, 3, 20, activation_function = tf.nn.relu)
predition = add_layer(l1, 20, 3, activation_function = None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition), reduction_indices=[1]))#一列
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 200 == 0:
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
    saver = tf.train.Saver()
    saver.save(sess, model_path)