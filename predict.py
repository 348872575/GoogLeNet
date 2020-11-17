from model import GoogLeNet
from PIL import Image
import numpy as np
import json
# import matplotlib.pyplot as plt
import codecs
import csv
import os
import tensorflow as tf

path = os.getcwd()
test_path = os.path.join(path,'Image_Classification','test')
test_num = len(os.listdir(test_path))

im_height = 224
im_width = 224
aux_logits = True

# 读class_indict文件
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)
model = GoogLeNet(class_num=6, aux_logits=aux_logits)  # 重新构建网络
model.summary()
model.load_weights("./save_weights/myGoogLeNet.h5", by_name=True)  # 加载模型参数
# model.load_weights("./save_weights/myGoogLeNet.ckpt")  # ckpt format

results = []

for i in range(test_num):
    imgname = str(i)+'.jpg'
    img_path = os.path.join(path,'Image_Classification','test',imgname)
    # 读入图片
    img = Image.open(img_path)  # 这是我的路径，要根据自己的根目录来改
    # resize成224x224的格式
    img = img.resize((im_width, im_height))
    # plt.imshow(img)
    # 对原图标准化处理
    img = ((np.array(img) / 255.) - 0.5) / 0.5
    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img, 0))

# model = GoogLeNet(class_num=6, aux_logits=False)  # 重新构建网络
# model.summary()
# model.load_weights("./save_weights/myGoogLenet.h5", by_name=True)  # 加载模型参数
# # model.load_weights("./save_weights/myGoogLeNet.ckpt")  # ckpt format
    result = model.predict(img)
    if aux_logits == False:
        predict_class = np.argmax(result)
        # print('预测出的类别是：', class_indict[str(predict_class)])  # 打印显示出预测类别
        final_result = class_indict[str(predict_class)]
        results.append([i, final_result])
    elif aux_logits == True:
        aux1_result = np.argmax(result[0])
        aux2_result = np.argmax(result[1])
        aux3_result = np.argmax(result[2])
        all = [aux1_result,aux2_result,aux3_result]
        predict_class = max(all, key=all.count)
        final_result = class_indict[str(predict_class)]
        results.append([i, final_result])

    # # print(result)
    # predict_class = np.argmax(result)
    # result =class_indict[str(predict_class)]
    # results.append([i,result])
    # # print('预测出的类别是：', class_indict[str(predict_class)])  # 打印显示出预测类别

def data_write_csv(file_name, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name,'w+','utf-8')#追加
    writer = csv.writer(file_csv, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("保存文件成功，处理结束")
data_write_csv('./answer.csv',results)
# plt.show()