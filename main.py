#! /usr/bin/python3
# -*- coding: utf8 -*-

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy
import math

import tensorflow as tf
import tensorlayer as tl
from model import Generator
from utils import *
from config import config, log_config
from PIL import Image

import re
import glob

###====================== HYPER-PARAMETERS ===========================###
## batch size
sample_batch_size = config.TRAIN.sample_batch_size
## Adam
batch_size = config.TRAIN.batch_size
learning_rate = config.TRAIN.learning_rate
## training
n_epoch = config.TRAIN.n_epoch
## paths
samples_path = config.samples_path
checkpoint_path = config.checkpoint_path
valid_hr_img_path = config.VALID.hr_img_path
train_hr_img_path = config.TRAIN.hr_img_path
eval_img_name_regx = config.VALID.eval_img_name_regx
eval_img_path = config.VALID.eval_img_path
save_file_format = config.save_file_format

ni = int(np.sqrt(sample_batch_size))


def load_deep_file_list(path=None, regx='\.npz', recursive=True, printable=True):
    if path == False:
        path = os.getcwd()
    pathStar = path + '**'
    file_list = glob.glob(pathStar, recursive=recursive)
    return_list = []
    for idx, f in enumerate(file_list):
        if re.search(regx, f):
            fShort = f.replace(path,'')
            return_list.append(fShort)
    if printable:
        print('Match file list = %s' % return_list)
        print('Number of files = %d' % len(return_list))
    return return_list


def train():
    ## create folders to save result images and trained model
    save_dir_generated = samples_path + "generated"
    tl.files.exists_or_mkdir(save_dir_generated)
    tl.files.exists_or_mkdir(checkpoint_path)

    ###====================== PRE-LOAD DATA ===========================###
    valid_hr_img_list = sorted(tl.files.load_file_list(path=valid_hr_img_path, regx='.*\.(bmp|png|webp|jpg)', printable=False))

    ###========================== DEFINE MODEL ============================###
    ## train inference
    sample_t_image = tf.compat.v1.placeholder('float32', [sample_batch_size, 96, 96, 3], name='sample_t_image_input_to_generator')
    t_image = tf.compat.v1.placeholder('float32', [batch_size, 96, 96, 3], name='t_image_input_to_generator')
    t_target_image = tf.compat.v1.placeholder('float32', [batch_size, 384, 384, 3], name='t_target_image')

    net_g = Generator(t_image, is_train=True, reuse=False)

    net_g.print_params(False)
    net_g.print_layers()

    ## test inference
    net_g_test = Generator(sample_t_image, is_train=False, reuse=True)

    # ###========================== DEFINE TRAIN OPS ==========================###

    # Loss
    mae_loss = tf.reduce_mean(tf.map_fn(tf.abs, t_target_image - net_g.outputs))
    edge_loss = tf.reduce_mean(tf.map_fn(tf.abs, tf.image.sobel_edges(t_target_image) - tf.image.sobel_edges(net_g.outputs)))
    g_loss = mae_loss + edge_loss

    with tf.variable_scope('learning_rate'):
        learning_rate_var = tf.Variable(learning_rate, trainable=False)

    g_vars = tl.layers.get_variables_with_name('Generator', True, True)

    ## Optimizer
    g_optim = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate_var).minimize(g_loss, var_list=g_vars)

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.variables_initializer(tf.global_variables()))
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_path + 'g.npz', network=net_g)

    ###============================= TRAINING ===============================###
    sample_imgs = tl.prepro.threading_data(valid_hr_img_list[0:sample_batch_size], fn=get_imgs_fn, path=valid_hr_img_path)
    sample_imgs_384 = tl.prepro.threading_data(sample_imgs, fn=crop_sub_imgs_fn, is_random=False)
    print('sample HR sub-image:', sample_imgs_384.shape, sample_imgs_384.min(), sample_imgs_384.max())
    sample_imgs_96 = tl.prepro.threading_data(sample_imgs_384, fn=downsample_fn)
    print('sample LR sub-image:', sample_imgs_96.shape, sample_imgs_96.min(), sample_imgs_96.max())
    save_images(sample_imgs_96, [ni, ni], save_file_format, save_dir_generated + '/_train_sample_96')
    save_images(sample_imgs_384, [ni, ni], save_file_format, save_dir_generated + '/_train_sample_384')

    ###========================= train =========================###
    sess.run(tf.assign(learning_rate_var, learning_rate))
    for epoch in range(0, n_epoch + 1):
        epoch_time = time.time()
        total_g_loss, total_mae_loss, total_edge_loss, step = 0, 0, 0, 0

        train_hr_img_list = load_deep_file_list(path=train_hr_img_path, regx='.*\.(bmp|png|webp|jpg)', recursive=True, printable=False)
        random.shuffle(train_hr_img_list)

        list_length = len(train_hr_img_list)
        print("Number of images: %d" % (list_length))

        if list_length % batch_size != 0:
            train_hr_img_list += train_hr_img_list[0:batch_size - list_length % batch_size:1]

        list_length = len(train_hr_img_list)
        print("Length of list: %d" % (list_length))
        n_step = list_length / batch_size

        for idx in range(0, list_length, batch_size):
            step_time = time.time()
            b_imgs_list = train_hr_img_list[idx : idx + batch_size]
            b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=train_hr_img_path)
            b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_data_augment_fn, is_random=True)
            b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)
            b_imgs_384 = tl.prepro.threading_data(b_imgs_384, fn=rescale_m1p1)

            ## update G
            errM, errE, errG, _ = sess.run([mae_loss, edge_loss, g_loss, g_optim], {t_image: b_imgs_96, t_target_image: b_imgs_384})
            print("Epoch[%2d/%2d] Step[%4d/%4d] time: %4.2fs g_loss: %.8f mae_loss: %.8f edge_loss: %.8f" %
                  (epoch, n_epoch, step, n_step,  time.time() - step_time, errG, errM, errE))
            total_g_loss += errG
            total_mae_loss += errM
            total_edge_loss += errE
            step += 1

        log = ("[*] Epoch[%2d/%2d] time: %4.2fs g_loss: %.8f mae_loss: %.8f edge_loss: %.8f" %
            (epoch, n_epoch, time.time() - epoch_time, total_g_loss / n_step, total_mae_loss / n_step, total_edge_loss / n_step))
        print(log)

        ## quick evaluation on train set
        out = sess.run(net_g_test.outputs, {sample_t_image: sample_imgs_96})
        print("[*] save images")
        save_images(out, [ni, ni], save_file_format, save_dir_generated + '/train_%d' % epoch)

        ## save model
        tl.files.save_npz(net_g.all_params, name=checkpoint_path + 'g.npz', sess=sess)


def evaluate():
    ## create folders to save result images
    save_dir = samples_path + "evaluate"
    tl.files.exists_or_mkdir(save_dir)

    ###========================== DEFINE MODEL ============================###
    eval_img_name_list = load_deep_file_list(path=eval_img_path, regx=eval_img_name_regx, recursive=False, printable=False)
    print(eval_img_name_list)
    valid_lr_img = get_imgs_fn(eval_img_name_list[0], eval_img_path) # if you want to test your own image
    valid_lr_img = rescale_m1p1(valid_lr_img)

    size = valid_lr_img.shape
    t_image = tf.compat.v1.placeholder('float32', [1, None, None, 3], name='input_image')

    net_g = Generator(t_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.variables_initializer(tf.global_variables()))
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_path + 'g.npz', network=net_g)

    ###======================= EVALUATION =============================###
    start_time = time.time()
    out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
    print("took: %4.4fs" % (time.time() - start_time))

    print("LR size: %s /  generated HR size: %s" % (size, out.shape))
    print("[*] save images")
    out = (out + 1) * 127.5 # rescale to [0, 255]
    out_uint8 = out.astype('uint8')
    save_img_fn(out_uint8[0], save_file_format, save_dir + '/valid_gen')

    out_bicu = (valid_lr_img + 1) * 127.5 # rescale to [0, 255]
    out_bicu = np.array(Image.fromarray(np.uint8(out_bicu)).resize((size[1] * 4, size[0] * 4), Image.BICUBIC))
    out_bicu_uint8 = out_bicu.astype('uint8')
    save_img_fn(out_bicu_uint8, save_file_format, save_dir + '/valid_bicubic')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', help='train, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'train':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")
