from aip import AipFace
import json
import base64
from PIL import Image
import os
import pandas as pd
from pandas import Series, DataFrame
import time
import csv


# 定义常量 (Define the user info for Baidu Face Recognition API)
APP_ID = '15016973'
API_KEY = 'mvZk1uBubsGKw0WPecVpLGui'
SECRET_KEY = 'gAVujH1tuOgPbtyNn1V9asZAlOkbD8cL'

# 设置人脸识别参数（Set the constant for the API）
options = {}
options["face_field"] = "landmark"
options["max_face_num"] = 1
options["face_type"] = "LIVE"

options1 = {}
options1["face_field"] = "quality"
options1["max_face_num"] = 1
options1["face_type"] = "LIVE"

nameList = []
kouhongrgb = []
skinColorRgb = []


def shelterValid(dic):
    valid = 1
    if(dic['chin_contour'] > 0.6):
        valid = 0 
    if(dic['nose'] > 0.7 or dic['mouth'] > 0.7):
        valid = 0
    if(dic['left_cheek'] > 0.6 or dic['right_cheek'] > 0.6):
        valid = 0
    if valid == 0:
        return False
    else:
        return True


def getAverage(x, y):
    return (x + y) / 2

# 标准化文件名(Normalize the filenames for iteration)


def listDir(rootDir):
    for filename in os.listdir(rootDir):
        pathname = os.path.join(rootDir, filename)
        nameList.append(pathname)


# 初始化AipFace对象(Initialize the objective of aipFace)
aipFace = AipFace(APP_ID, API_KEY, SECRET_KEY)

# 读取图片(Initialize the directory; Pictures with girl's faces are in the directory)
listDir('./test1')

fobj = open('./data2.csv', 'a+')
inputin = csv.writer(fobj,lineterminator='\n')

# 对文件进行迭代(Iterate the picture files in the directory)
for i in nameList[0: 30]:
    time.sleep(0.5)
    with open(i, "rb") as f:
        base64_data = base64.b64encode(f.read())

    print(str(nameList.index(i)) + '/' + str(len(nameList)))

    result = aipFace.detect(str(base64_data)[2:], "BASE64", options) 
    result1 = aipFace.detect(str(base64_data)[2:], "BASE64", options1)
    
    print(result['error_code'], result1['error_code'])

    print(result['error_code'])
    if(result['error_code'] == 0 and result1['error_code'] == 0 and abs(result['result']['face_list'][0]['angle']['yaw']) < 20 and abs(result['result']['face_list'][0]['angle']['pitch']) < 20 and abs(result['result']['face_list'][0]['angle']['roll']) < 20 and shelterValid(result1['result']['face_list'][0]['quality']['occlusion']) and result1['result']['face_list'][0]['quality']['blur'] < 0.9 and result1['result']['face_list'][0]['quality']['completeness'] == 1):
        print(i)
        # landmark72: 百度人脸识别API中获取人脸72个点的函数(landmark72: Function in the API that can get 72 points of a people's face）
        landmark72 = result['result']['face_list'][0]['landmark72']

        im = Image.open(i)
        pix = im.convert('RGBA')

    # 计算口红颜色（取嘴唇上的六个点然后取rgb平均值）
    # Calculate the color of the lipstick (Get 6 points from the lipstick and calculate the average RGB)
        x_1, y_1 = landmark72[59]['x'], landmark72[59]['y']
        x_2, y_2 = landmark72[66]['x'], landmark72[66]['y']
        x1 = getAverage(x_1, x_2)
        y1 = getAverage(y_1, y_2)

        r1, g1, b1, a1 = pix.getpixel((x1, y1))

        x_1, y_1 = landmark72[60]['x'], landmark72[60]['y']
        x_2, y_2 = landmark72[67]['x'], landmark72[67]['y']
        x2 = getAverage(x_1, x_2)
        y2 = getAverage(y_1, y_2)

        r2, g2, b2, a2 = pix.getpixel((x2, y2))

        x_1, y_1 = landmark72[61]['x'], landmark72[61]['y']
        x_2, y_2 = landmark72[68]['x'], landmark72[68]['y']
        x3 = getAverage(x_1, x_2)
        y3 = getAverage(y_1, y_2)

        r3, g3, b3, a3 = pix.getpixel((x3, y3))

        x_1, y_1 = landmark72[71]['x'], landmark72[71]['y']
        x_2, y_2 = landmark72[65]['x'], landmark72[65]['y']
        x4 = getAverage(x_1, x_2)
        y4 = getAverage(y_1, y_2)

        r4, g4, b4, a4 = pix.getpixel((x4, y4))

        x_1, y_1 = landmark72[70]['x'], landmark72[70]['y']
        x_2, y_2 = landmark72[64]['x'], landmark72[64]['y']
        x5 = getAverage(x_1, x_2)
        y5 = getAverage(y_1, y_2)

        r5, g5, b5, a5 = pix.getpixel((x5, y5))

        x_1, y_1 = landmark72[69]['x'], landmark72[69]['y']
        x_2, y_2 = landmark72[63]['x'], landmark72[63]['y']
        x6 = getAverage(x_1, x_2)
        y6 = getAverage(y_1, y_2)

        r6, g6, b6, a6 = pix.getpixel((x6, y6))

        r = int((r1 + r2 + r3 + r4 + r5 + r6) / 6)
        g = int((g1 + g2 + g3 + g4 + g5 + g6) / 6)
        b = int((b1 + b2 + b3 + b4 + b5 + b6) / 6)
        a = int((a1 + a2 + a3 + a4 + a5 + a6) / 6)
        rgba = (r, g, b, a)
        
        '''
        q = 500
        w = 500
        imn = Image.new("RGB", (q, w))
        for i in range(0, q):
            for j in range(0, w):
                imn.putpixel((i, j), (r, g, b))
        imn.show()
        imn.save("xx.jpg")
        '''
        kouhongrgb.append(rgba)
        datas = []
        datas.append(rgba)

    # 提取肤色rgb
    # Get the skin color RGB with the same method

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
        skinColorRgb.append(rgba)
        datas.append(rgba)
        
        change = datas[1]
        datas[1] = datas[0]
        datas[0] = change
        
        inputin.writerow(datas)


fobj.close()
s = Series(kouhongrgb, index=skinColorRgb)

s.to_csv("dataFile5.csv")
