from aip import AipFace
import base64
from PIL import Image
import tensorflow as tf
import numpy as np
import csv

imgFile = 'test.jpg'

def getAverage(x, y):
    return (x + y) / 2

def shelterValid(dic):
    valid = 1
    if(dic['chin_contour'] > 0.6):
        valid = 0 
    if(dic['nose'] > 0.7 or dic['mouth'] > 0.7):
        valid = 0
    if(dic['left_cheek'] > 0.8 or dic['right_cheek'] > 0.8):
        valid = 0
    if valid == 0:
        return False
    else:
        return True

""" 你的 APPID AK SK """
APP_ID = '15016973'
API_KEY = 'mvZk1uBubsGKw0WPecVpLGui'
SECRET_KEY = 'gAVujH1tuOgPbtyNn1V9asZAlOkbD8cL'

client = AipFace(APP_ID, API_KEY, SECRET_KEY)

faceImage = open(imgFile,'rb') #二进制方式打开图文件
faceBase64 = str(base64.b64encode(faceImage.read())) #读取文件内容，转换为base64编码 
faceBase64 = faceBase64[2:] 
faceImage.close()


imageType = "BASE64"

options = {}
options["face_field"] = "landmark"
options["max_face_num"] = 1
options["face_type"] = "LIVE"

result = client.detect(faceBase64, imageType, options)

print(result)

landmark72 = result['result']['face_list'][0]['landmark72']

im = Image.open(imgFile)
pix = im.convert('RGBA')

x_1, y_1 = landmark72[48]['x'], landmark72[48]['y']
x_2, y_2 = landmark72[55]['x'], landmark72[55]['y']
x1 = getAverage(x_1, x_2)
y1 = getAverage(y_1, y_2)

r1, g1, b1, a1 = pix.getpixel((x1, y1))

x_1, y_1 = landmark72[1]['x'], landmark72[1]['y']
x_2, y_2 = landmark72[49]['x'], landmark72[49]['y']
x2 = getAverage(x_1, x_2)
y2 = getAverage(y_1, y_2)

r2, g2, b2, a2 = pix.getpixel((x2, y2))

x_1, y_1 = landmark72[54]['x'], landmark72[54]['y']
x_2, y_2 = landmark72[11]['x'], landmark72[11]['y']
x3 = getAverage(x_1, x_2)
y3 = getAverage(y_1, y_2)

r3, g3, b3, a3 = pix.getpixel((x3, y3))

x_1, y_1 = landmark72[2]['x'], landmark72[2]['y']
x_2, y_2 = landmark72[50]['x'], landmark72[50]['y']
x4 = getAverage(x_1, x_2)
y4 = getAverage(y_1, y_2)

r4, g4, b4, a4 = pix.getpixel((x4, y4))

x_1, y_1 = landmark72[10]['x'], landmark72[10]['y']
x_2, y_2 = landmark72[53]['x'], landmark72[53]['y']
x5 = getAverage(x_1, x_2)
y5 = getAverage(y_1, y_2)

r5, g5, b5, a5 = pix.getpixel((x5, y5))

x_1, y_1 = landmark72[2]['x'], landmark72[2]['y']
x_2, y_2 = landmark72[58]['x'], landmark72[58]['y']
x6 = getAverage(x_1, x_2)
y6 = getAverage(y_1, y_2)

r6, g6, b6, a6 = pix.getpixel((x6, y6))

x_1, y_1 = landmark72[10]['x'], landmark72[10]['y']
x_2, y_2 = landmark72[62]['x'], landmark72[62]['y']
x7 = getAverage(x_1, x_2)
y7 = getAverage(y_1, y_2)

r7, g7, b7, a7 = pix.getpixel((x7, y7))

x_1, y_1 = landmark72[3]['x'], landmark72[3]['y']
x_2, y_2 = landmark72[58]['x'], landmark72[58]['y']
x8 = getAverage(x_1, x_2)
y8 = getAverage(y_1, y_2)

r8, g8, b8, a8 = pix.getpixel((x8, y8))

x_1, y_1 = landmark72[9]['x'], landmark72[9]['y']
x_2, y_2 = landmark72[62]['x'], landmark72[62]['y']
x9 = getAverage(x_1, x_2)
y9 = getAverage(y_1, y_2)

r9, g9, b9, a9 = pix.getpixel((x9, y9))

x_1, y_1 = landmark72[4]['x'], landmark72[4]['y']
x_2, y_2 = landmark72[65]['x'], landmark72[65]['y']
x10 = getAverage(x_1, x_2)
y10 = getAverage(y_1, y_2)

r10, g10, b10, a10 = pix.getpixel((x10, y10))

x_1, y_1 = landmark72[8]['x'], landmark72[8]['y']
x_2, y_2 = landmark72[63]['x'], landmark72[63]['y']
x11 = getAverage(x_1, x_2)
y11 = getAverage(y_1, y_2)

r11, g11, b11, a11 = pix.getpixel((x11, y11))

x_1, y_1 = landmark72[5]['x'], landmark72[5]['y']
x_2, y_2 = landmark72[64]['x'], landmark72[64]['y']
x12 = getAverage(x_1, x_2)
y12 = getAverage(y_1, y_2)

r12, g12, b12, a12 = pix.getpixel((x12, y12))

x_1, y_1 = landmark72[6]['x'], landmark72[6]['y']
x_2, y_2 = landmark72[64]['x'], landmark72[64]['y']
x13 = getAverage(x_1, x_2)
y13 = getAverage(y_1, y_2)

r13, g13, b13, a13 = pix.getpixel((x13, y13))

x_1, y_1 = landmark72[7]['x'], landmark72[7]['y']
x_2, y_2 = landmark72[64]['x'], landmark72[64]['y']
x14 = getAverage(x_1, x_2)
y14 = getAverage(y_1, y_2)

r14, g14, b14, a14 = pix.getpixel((x14, y14))

r = int((r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 +
         r9 + r10 + r11 + r12 + r13 + r14) / 14)
g = int((g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8 +
         g9 + g10 + g11 + g12 + g13 + g14) / 14)
b = int((b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 +
         b9 + b10 + b11 + b12 + b13 + b14) / 14)
a = int((a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 +
         a9 + a10 + a11 + a12 + a13 + a14) / 14)

rgba = (r, g, b, a)
print(rgba)

model_path = './tensorflow_net/model.ckpt'

x_data = np.array([[r, g, b]])

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
    #count = 0
    for p in sess.run(predition, feed_dict={xs: x_data}):
        hight = 500
        weight = 500
        imn = Image.new("RGB", (hight, weight))
        for i in range(0, hight):
            for j in range(0, weight):
                imn.putpixel((i, j), (p[0], p[1], p[2]))
        imn.save('./test_rgb.jpg')
        imn.show()
        #count += 1