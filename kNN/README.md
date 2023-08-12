# kNN算法实现

## 文件内容

实现了基于Cifar10与MINST数据集的kNN算法，并比较了不同k值之间的区别。

- `Database`目录存放数据集
- `Doc`目录存放任务要求与报告
- `Project`目录存放代码
- `dataloader.py`实现数据处理与读取
- `kNN-Cifar10.py`和`kNN-MNIST.py`实现了对两个数据集的kNN算法。

## 最终数据

- 对于`Cifar10`，k = 10 时，准确率最高，为57%。
- 对于`MINST`, k = 3 时，准确率最高，为97%。

根据任务要求，需要实现不同的k值选择和不同的距离，这里采用了`曼哈顿距离`和`欧式距离`进行计算，具体的优劣分析以及代码分析见`Doc`目录下的报告。

## 前置需求

基于Python版本3.8，只要是Python3均可正常运行。

文档需要的库包括`numpy`与`matplotlib.pyplot。

直接运行`kNN-Cifar10.py`和`kNN-MNIST.py` 即可。