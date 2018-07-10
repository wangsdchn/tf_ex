#!usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:WSD 
@file: t_.py 
@time: 6:16 PM 
"""
import tensorflow as tf
import numpy as np
import input_data
import alexnet
from PIL import Image
import matplotlib.pyplot as plt


def get_one_image(train):
    '''Randomly pick one image from training data
    Return: ndarray
    '''
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]
    print(img_dir)

    image = Image.open(img_dir)
    plt.imshow(image)
    image = image.resize([224, 224])
    image = np.array(image)
    return image


def evaluate_one_image():
    '''Test one image against the saved models and parameters
    '''

    # you need to change the directories to yours.
    # train_dir = '/home/kevin/tensorflow/cats_vs_dogs/data/train/'
    train_dir = './dataset/train/'
    train, train_label = input_data.get_files(train_dir)
    image_array = get_one_image(train)

    with tf.Graph().as_default():
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 224, 224, 3])
        logit = alexnet.inference(image)

        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[224, 224, 3])

        # you need to change the directories to yours.
        # logs_train_dir = '/home/kevin/tensorflow/cats_vs_dogs/logs/train/'
        logs_train_dir = './logs'

        saver = tf.train.Saver()

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            if max_index == 0:
                print('This is a cat with possibility %.6f' % prediction[:, 0])
            else:
                print('This is a dog with possibility %.6f' % prediction[:, 1])


if __name__ == '__main__':
    evaluate_one_image()

