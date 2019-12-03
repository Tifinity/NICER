'''
Time: 2018/12/10
Author: Tifinity
'''
import tensorflow as tf
import numpy as np
import csv
from PIL import Image

model_path = './tensorflow_net/model.ckpt'
test_filename = './test_data.csv'

x_data = np.array([[0, 0, 0]])
#从文件中读取测试数据
with open(test_filename) as f:
    reader = csv.reader(f)
    st = list(reader)
    print(len(st))
    for i in st:
        x = []
        count = 0
        i[0] = i[0].strip('()').replace(' ', '')
        s = ""
        for j in i[0]:
            if(j == ','):
                x.append(int(s))
                count += 1
                s = ""
            else:
                s += j
        x_data = np.r_[x_data, np.array([[x[0], x[1], x[2]]])]
x_data = np.delete(x_data, 0, 0)

#重建神经网络结构
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
l1 = add_layer(xs, 3, 20, activation_function = tf.nn.relu)
predition = add_layer(l1, 20, 3, activation_function = None)

#读取已保存的参数
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, model_path)
    print(sess.run(predition, feed_dict={xs: x_data}))
    #输出rgb色块的图片
    count = 0
    for p in sess.run(predition, feed_dict={xs: x_data}):
        hight = 500
        weight = 500
        imn = Image.new("RGB", (hight, weight))
        for i in range(0, hight):
            for j in range(0, weight):
                imn.putpixel((i, j), (p[0], p[1], p[2]))
        imn.save('./test_rgb/' + str(count) + '.jpg')
        count += 1