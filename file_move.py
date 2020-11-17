import os
import shutil
from random import shuffle
import csv

"将官方给的数据集按比例分成train和valid文件夹，并按照类别再细分成不同文件夹，以便利用tensorflow生成训练数据集"

validation_split = 0.2

csv_list =[]
dir = os.getcwd()
old_path = os.path.join(dir, 'Image_Classification', 'train')
csv_path = os.path.join(dir, 'Image_Classification', 'train.csv')

train_path = {'buildings': './data/train/buildings', 'forest': './data/train/forest', 'glacier': './data/train/glacier',
            'mountain': './data/train/mountain', 'sea': './data/train/sea', 'street': './data/train/street'}

val_path = {'buildings': './data/valid/buildings', 'forest': './data/valid/forest', 'glacier': './data/valid/glacier',
            'mountain': './data/valid/mountain', 'sea': './data/valid/sea', 'street': './data/valid/street'}

# with open('./Image_Classification/train.csv') as file:
#     reader = csv.reader(file)
#     head = next(reader)
#     for row in reader:
#         old_filepath = os.path.join(old_path, row[0])
#         shutil.copy(old_filepath, new_path[row[1]])


with open(csv_path) as file:
    reader = csv.reader(file)
    head = next(reader)
    for row in reader:
        csv_list.append(row)

print("the total number of images is:",len(csv_list))
validation_num = int(len(csv_list)*validation_split)
train_num = int(len(csv_list)-validation_num)
print("validation_num:",validation_num)
print("train_num:",train_num)

shuffle(csv_list)
valid_list = csv_list[:validation_num]
train_list = csv_list[validation_num:]

for valid in valid_list:
    old_filepath = os.path.join(old_path, valid[0])
    shutil.copy(old_filepath, val_path[valid[1]])

for train in train_list:
    old_filepath = os.path.join(old_path, train[0])
    shutil.copy(old_filepath, train_path[train[1]])
