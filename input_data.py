# -*- coding: utf-8 -*-
"""
Created on Thur Dec 14 17:48 2017
@author: WSD
"""
import tensorflow as tf
import numpy as np
import os


# 读取文件名和标签
def get_files(file_dir):
    # file_dir 图片存放路径
    # 返回shuffle 后的图片和标签

    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    # load image path and label
    for file in os.listdir(file_dir):
        name = file.split('.')
        if name[0] == 'cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    print("There are %d cats and %d dogs" % (len(cats), len(dogs)))

    # shuffle
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))
    temp = np.array([image_list, label_list])
    temp = temp.T
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    return image_list, label_list


# 生成相同大小的批次
def get_batch(image, label, image_w, image_h, batch_size, capacity):
    # image, label: 要生成batch的图像和标签list
    # image_W, image_H: 图片的宽高
    # batch_size: 每个batch有多少张图片
    # capacity: 队列容量
    # return: 图像和标签的batch
    # 将python list 转换为tf可识别的格式
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # 生成队列
    input_queue = tf.train.slice_input_producer([image, label])
    image_contents = tf.read_file(input_queue[0])
    label = input_queue[1]
    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, image_w, image_h)
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)  # 标准化数据
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=64, capacity=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch


if __name__ == '__main__':
    BATCH_SIZE = 2
    CAPACITY = 256
    IMG_W = 208
    IMG_H = 208

    train_dir = "./dataset/train/"
    test_dir = "./dataset/test/"
    train_list, train_label_list = get_files(train_dir)
    test_list, test_label_list = get_files(test_dir)
