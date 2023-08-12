import pickle
import numpy as np
import os
import platform
import struct


def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = load_pickle(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte


def load_MINST(file_name):
    # 读取整个文件进入缓冲区
    file_content = open(file_name, "rb").read()
    # 读取文件头信息，共有16个字节,从offset=0开始读取
    magic_number, num_images, num_rows, num_cols = struct.unpack_from('>IIII', file_content, 0)
    offset = struct.calcsize('>IIII')
    # 读取之后的图片信息
    image_size = num_rows * num_cols
    fmt_image = '>' + str(image_size) + 'B'  # '>'表示大端，'B'表示integer(1个字节)
    images = np.empty((num_images, image_size))  # 新建一个大的矩阵存放图片信息
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, file_content, offset))
        offset += struct.calcsize(fmt_image)
    return images

def load_label(file_name):
    # 读取整个文件进入缓冲区
    file_content = open(file_name, "rb").read()
    # 读取文件头信息，共有8个字节,从offset=0开始读取
    magic_number, labels_num = struct.unpack_from('>II', file_content, 0)
    offset = struct.calcsize('>II')
    # 读取之后的标签信息
    fmt_label = '>' + str(labels_num) + 'B'
    labels = np.array(struct.unpack_from(fmt_label, file_content, offset))
    return labels
