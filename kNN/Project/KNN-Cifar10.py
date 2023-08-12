import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(np.int32)
    return X, y


def compute_distances(X, X_train):
    num_test = X.shape[0]
    num_train = X_train.shape[0]
    dists = np.zeros((num_test, num_train))

    # 计算欧氏距离
    dists = np.sqrt(
        np.sum(np.square(X_train), axis=1) + np.sum(np.square(X), axis=1)[:, np.newaxis] - 2 * np.dot(X, X_train.T))

    return dists

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

def predict_labels(dists, y_train, k=1):
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)

    for i in range(num_test):
        closest_y = []
        sorted_dist = np.argsort(dists[i])
        closest_y = list(y_train[sorted_dist[0:k]])

        # 统计最近邻的标签
        y_pred[i] = np.argmax(np.bincount(closest_y))

    return y_pred


def accuracy(y_pred, y_true):
    correct = np.sum(y_pred == y_true)
    total = y_pred.shape[0]
    return correct / total


def mnist_knn():
    # 加载MNIST数据集
    X, y = load_mnist()

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 调整数据形状
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    # 计算测试集和训练集之间的距离
    dists = compute_distances(X_test, X_train)

    # 在测试集上进行预测
    y_pred = predict_labels(dists, y_train, k=5)

    # 计算准确率
    acc = accuracy(y_pred, y_test)
    print("Accuracy: {:.2%}".format(acc))


if __name__ == "__main__":
    mnist_knn()