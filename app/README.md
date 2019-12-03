# NICE小组复赛提交材料说明

## .py代码说明

### face_rgb.py
批量分析爬取到文件夹中的化妆人脸图片，得到其中所有人脸的面部平均RGB数值、嘴唇平均RGB数值，并存入.csv文档中，用于之后的神经网络训练。
### net_train.py
读取数据，用tensorflow框架训练神经网络。
### net_test.py
使用训练好的神经网络读入二十条人脸面部平均RGB数值，得到输出（颜色图片，代表最适口红颜色）。
### one_face_to_color.py
展示时用Demo程序，传入一张人脸图片读取其面部平均RGB数值，传入神经网络，得到输出（颜色图片，代表最适口红颜色）。

## .csv文件说明

### train_data.csv
用于训练神经网络的数据集。
### test_data.csv
用于测试训练好的神经网络的测试集。

## 文件夹说明

### matlab_net
使用同样数据集，通过matlab训练的神经网络。
### tensorflow_net
通过tensorflow训练的神经网络的参数。
### test_rgb
测试集经过tensorflow神经网络后得到的输出（颜色图片，代表最适口红颜色）。

## .jpg图片说明

### test.jpg

输入到one_face_to_color.py中，输出据此人脸分析得到的最适口红颜色。
### test_rgb.jpg
test.jpg的输出。

