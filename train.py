#!usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:WSD 
@file: train.py 
@time: 1:05 PM 
"""

import os
import numpy as np
import tensorflow as tf
import input_data
import vgg
import alexnet


N_CLASSES = 2
IMG_W = 227  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 227
BATCH_SIZE = 32
CAPACITY = 64
MAX_STEP = 20000  # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001  # with current parameters, it is suggested to use learning rate<0.0001

train_layers = {'fc6', 'fc7', 'fc8'}

# %%
def run_training():
    train_dir = './dataset/train/'  # My dir--20170727-csq
    logs_train_dir = './logs/'
    train, train_label = input_data.get_files(train_dir)

    train_batch, train_label_batch = input_data.get_batch(train,
                                                          train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE,
                                                          CAPACITY)

    train_logits = alexnet.inference(train_batch)
    var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
    train_loss = alexnet.losses(train_logits, train_label_batch)
    train_op = alexnet.trainning(train_loss, var_list, learning_rate)
    train__acc = alexnet.evaluation(train_logits, train_label_batch)

    summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        saver = tf.train.Saver()

        alexnet.load_initial_weights('bvlc_alexnet.npy', train_layers, sess)
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break
                _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])
                
                sess.run([train_batch,train_label_batch])
                if step % 10 == 0:
                    print(train_logits,train_label_batch)
                    print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                    summary_str = sess.run(summary_op)
                    train_writer.add_summary(summary_str, step)

                if step % 2000 == 0 or (step + 1) == MAX_STEP:

                    checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    run_training()

