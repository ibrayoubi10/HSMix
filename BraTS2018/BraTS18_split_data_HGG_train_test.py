
import os
import random
import shutil
from shutil import copy2


"""制作类别图像的训练集，和测试集所需要的文件夹，每个文件夹含二级路径"""
def mkTotalDir(data_path):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    # dic=['train','val','test']
    # for i in range(0,3):
    #     current_path=data_path+dic[i]+'/'
    #     #这个函数用来判断当前路径是否存在，如果存在则创建失败，如果不存在则可以成功创建
    #     isExists=os.path.exists(current_path)
    #     if not isExists:
    #         os.makedirs(current_path)
    #         print('successful '+dic[i])
    #     else:
    #         print('is existed')
    # return

"""
source_path:原始多类图像的存放路径
train_path:训练集图像的存放路径
test_path:测试集图像的存放路径
train : test = 8 : 2
"""
"""
Updated to ignore hidden files like .DS_Store when generating txt files. 
author : Ibrahim
"""

def divideTrainValidationTest(source_path, train_path, test_path):
    # List only subdirectories (patients), ignore files like .DS_Store
    entries = [d for d in os.listdir(source_path)
               if os.path.isdir(os.path.join(source_path, d)) and not d.startswith(".")]

    random.shuffle(entries)

    split = int(0.2 * len(entries))
    test_image_list = entries[:split]
    train_image_list = entries[split:]

    def copy_patient(patient_id, dst_root):
        src_dir = os.path.join(source_path, patient_id)
        dst_dir = os.path.join(dst_root, patient_id)
        os.makedirs(dst_dir, exist_ok=True)

        for fname in os.listdir(src_dir):
            src_file = os.path.join(src_dir, fname)
            # copy only files (skip nested dirs if any)
            if os.path.isfile(src_file) and not fname.startswith("."):
                copy2(src_file, dst_dir)

    for pid in train_image_list:
        copy_patient(pid, train_path)

    for pid in test_image_list:
        copy_patient(pid, test_path)



""""生成测试集、验证集、测试集的txt文件"""
def generatetxt(train_path,test_path):

    files_train = os.listdir(train_path)
    files_test = os.listdir(test_path)

    train = open('./data/BraTS2018_split/train_HGG.txt', 'a')
    test = open('./data/BraTS2018_split/test_HGG.txt', 'a')

    for file in files_train:
        fileType = os.path.split(file)
        if fileType[1] == '.txt':
            continue
        name =file + ' '  +  '\n'
        train.write(name)
    for file in files_test:
        fileType = os.path.split(file)
        if fileType[1] == '.txt':
            continue
        name =file + ' '  +  '\n'
        test.write(name)

    train.close()
    test.close()


if __name__=='__main__':
    data_path = './data/BraTS2018_split'#划分以后的train.val.test图像文件夹的存放位置
    source_path = './data/MICCAI_BraTS_2018_Data_Training/HGG'#划分前所有图像文件夹的存放位置（文件夹的存储层级是一级，按照标签命名）
    train_path = './data/BraTS2018_split/train/HGG'#划分以后训练集图像对应的存放位置
    test_path = './data/BraTS2018_split/test/HGG'#划分以后测试集图像对应的存放位置

    mkTotalDir(data_path)#按照路径建立训练集/测试集/验证集划分后的文件夹
    mkTotalDir(train_path)
    mkTotalDir(test_path)


    divideTrainValidationTest(source_path,train_path, test_path)#整体列表打乱后划分，并将图像移动到对应文件夹
    generatetxt(train_path,test_path)#根据对应划分后的图片文件夹，生成对应的txt文件

