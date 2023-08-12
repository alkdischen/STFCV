import operator
import os
import numpy as np
import struct
import datetime
import dataloader
import matplotlib.pyplot as plt
from multiprocessing import Process, Manager, Lock



def save_error(error_test_count, error_lock, errors_list):
    with error_lock:  # 因为errors_list是存放在进程公共通信区的，所以需要加把锁
        errors_list.append(error_test_count)

def KNN(test_image, test_label, train_images, train_labels, k, process_index, test_number, process_number, error_lock,
        errors_list):
    for test_count in range(process_index, test_number, process_number):
        # 读取训练集的行数,即训练集有多少张图片
        train_images_num = train_images.shape[0]
        # 求距离：先tile函数将测试集拓展成与训练集相同维数的矩阵，计算测试样本与每一个训练样本的欧式距离
        # 通过欧式距离的大小来判断图片的相似度
        all_distances = (np.sum((np.tile(test_image[test_count], (train_images_num, 1)) - train_images) ** 2,
                                axis=1)) ** 0.5
        # 按all_distances中元素进行升序排序后得到其对应索引的列表
        sorted_distance_index = all_distances.argsort()
        # 选择距离最小的k个样本，看一下它们中大部分都是哪个数字的样本
        classCount = np.zeros((10), dtype=int)
        # 10代表有10个数字，元素值表示对应数字的样本在这k个样本中出现了几次
        for i in range(k):
            vote_label = train_labels[sorted_distance_index[i]]
            classCount[vote_label] += 1
        # 找出出现最多的数字样本，为预测值
        result_label = -1
        max_times = 0
        for i in range(10):
            if classCount[i] >= max_times:
                max_times = classCount[i]
                result_label = i
        if (result_label != test_label[test_count]):
            save_error(test_count, error_lock, errors_list)
            print('…………（错误！）', end='')
        print(' ')


def main():
    t1 = datetime.datetime.now()  # 计时开始
    k_values = range(1, 101)  # 测试的k值范围

    # 载入文件
    train_image = dataloader.load_MINST('..\\Database\\MNIST\\train-images.idx3-ubyte')
    train_label = dataloader.load_label('..\\Database\\MNIST\\train-labels.idx1-ubyte')
    test_image = dataloader.load_MINST('..\\Database\\MNIST\\t10k-images.idx3-ubyte')
    test_label = dataloader.load_label('..\\Database\\MNIST\\t10k-labels.idx1-ubyte')

    accuracies = []  # 存储准确率

    for k in k_values:
        # 运行KNN算法
        error_lock = Lock()
        errors_list = Manager().list()
        processes = []
        process_number = 4  # 指定进程数
        test_number = len(test_image)  # 测试样本数量
        for i in range(process_number):
            p = Process(target=KNN,
                        args=(test_image, test_label, train_image, train_label, k, i, test_number, process_number,
                              error_lock, errors_list))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        accuracy = (test_number - len(errors_list)) / test_number * 100
        accuracies.append(accuracy)

    # 绘制准确率图表
    plt.plot(k_values, accuracies)
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. k')
    plt.show()

    t2 = datetime.datetime.now()
    print('耗 时 = ', t2 - t1)


if __name__ == "__main__":
    main()
