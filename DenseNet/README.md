# DenseNet Pre

2023.8.20 

课程内容为`DenseNet`的讲解。

`slide.ppt`内存放了讲解`ppt`

`code`文件夹存放代码，实现了基于`DenseNet`的，对房间整洁与脏乱的二元判断。由于一些问题（懒得debug了），代码输出训练结果并不能画图，需要手动添加数据到`plt.py`内画图，可以交给gpt代劳。

直接运行`main.py`即可运行，运用了`sk-learn`和`pretrain`过程。可能会有一些路径错误，用的绝对路径，需要修改。

同时，输出的应该不包括梯度内容，但包括`train`和`val`的`loss`和`accuracy`。