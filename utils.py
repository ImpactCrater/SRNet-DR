import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
# from config import config, log_config
#
# img_path = config.TRAIN.img_path

import scipy
import numpy as np
from PIL import Image, ImageMath
import random
from io import BytesIO
from config import config

noise_level = config.TRAIN.noise_level

def rescale_m1p1(x):
    x = x / 127.5 - 1 # rescale to [－1, 1]
    return x

def get_imgs_fn(file_name, path):
    x = np.asarray(Image.open(path + file_name))
    return x

def save_img_fn(x, save_file_format, file_name):
    x = Image.fromarray(np.uint8(x))
    if save_file_format == '.webp':
        x.save(file_name + save_file_format, lossless = True, quality = 100, method = 6)
    else:
        x.save(file_name + save_file_format)

def save_images(images, size, save_file_format, image_path='_temp'):
    """Save multiple images into one single image.

    Parameters
    -----------
    images: numpy array
        (batch, w, h, c)
    size: list of 2 ints
        row and column number.
        number of images should be equal or less than size[0] * size[1]
    save_file_format: str
        '\.(bmp|png|webp|jpg)'
    image_path: str
        save path

    """
    if len(images.shape) == 3:  # Greyscale [batch, h, w] --> [batch, h, w, 1]
        images = images[:, :, :, np.newaxis]

    def merge(images, size):
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3), dtype=images.dtype)
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img

    def imsave(images, size, save_file_format, path):
        if np.max(images) <= 2 and np.min(images) < 0:
            images = ((images + 1) * 127.5).astype(np.uint8)

        return save_img_fn(merge(images, size), save_file_format, path)

    if len(images) > size[0] * size[1]:
        raise AssertionError("number of images should be equal or less than size[0] * size[1] {}".format(len(images)))

    return imsave(images, size, save_file_format, image_path)

def crop_sub_imgs_fn(img, is_random=True):
    img = crop(img, wrg=384, hrg=384, is_random=is_random)
    return img

def crop_data_augment_fn(img, is_random=True):
    min_size = img.shape[0] if img.shape[0] < img.shape[1] else img.shape[1]
    random_size = random.randrange(384, min_size)
    img = crop(img, wrg=random_size, hrg=random_size, is_random=is_random)
    img = Image.fromarray(np.uint8(img)).resize((384, 384), Image.BICUBIC)
    h, s, v = img.convert("HSV").split()
    random_value = random.randrange(-12, 12)
    h_shifted = h.point(lambda x: (x + random_value) % 255 if (x + random_value) % 255 >= 0 else 255 - (x + random_value))
    img = Image.merge("HSV", (h_shifted, s, v)).convert("RGB")
    img = np.array(img)
    return img

def downsample_fn(x):
    x = Image.fromarray(np.uint8(x)).resize((96, 96), Image.BICUBIC)
    q = random.randrange(noise_level, 101)
    img_file = BytesIO()
    x.save(img_file, 'webp', quality=q)
    x = Image.open(img_file)
    x = np.array(x) / 127.5 - 1
    return x
    
