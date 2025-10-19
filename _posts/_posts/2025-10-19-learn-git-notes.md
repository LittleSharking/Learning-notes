# 软件工程原理与实践实验报告

<center>姓名：李东旭  学号：23020007056</center>

| 姓名和学号         | 李东旭，23020007056                  |
| -------------------- | ------------------------------ |
| 课程 | 中国海洋大学25秋《软件工程原理与实践》 |
| 实验名称           | 实验3：卷积神经网络          |

## 一、实验内容

### 1. 代码练习
#### 1.1 MNIST数据集分类
##### 1.1.1 下载并打印数据
使用MNIST下载数据
<img width="1000" height="850" alt="6247d5a133ce5d8582ae2f2bb1d2e560" src="https://github.com/user-attachments/assets/5bf5048e-257f-4e03-917f-02eb1a140955" />

<img width="1000" height="900" alt="bb4a5598ffc44cb30b44d367e8993981" src="https://github.com/user-attachments/assets/3de8d144-cbec-4518-82c1-df6014d9a77c" />


##### 1.1.2 分别在小型全连接网络和卷积神经网络上训练
（1）小型全连接网络上
<img width="1000" height="1000" alt="1f4f61c73279651288de3fb2dff7b0e7" src="https://github.com/user-attachments/assets/22e50e58-d5c2-4524-a26e-70a2717d5fcb" />

（2）卷积网络
<img width="1000" height="900" alt="6c7aff9fb124ed7d6a14ca02c94167e5" src="https://github.com/user-attachments/assets/6fa35118-efcf-4208-8a4d-449762de1930" />

##### 1.1.3 打乱数据顺序后再分别训练
（1）小型全连接网络上
<img width="1000" height="750" alt="a0f387d8413fc0766cb7a13d0d4910d2" src="https://github.com/user-attachments/assets/28b36c89-2e7f-4fed-b533-6c48a7370216" />

（2）卷积网络
<img width="1000" height="700" alt="fcac2559cf1ddd3d79bd19bc25984199" src="https://github.com/user-attachments/assets/27b67246-0702-4c5e-a636-f2a61312eada" />

#### 1.2 CIFAR10 数据集分类
##### 1.2.1 加载并归一化 CIFAR10 数据
<img width="1000" height="1000" alt="f31fb4e804b0ce53d737b771183e1e70" src="https://github.com/user-attachments/assets/ad872ff3-2d29-412e-8e13-bf6ce3707ce8" />

##### 1.2.2 卷积网络训练
<img width="1000" height="1000" alt="076fe42ff23317addf378a9925a6efda" src="https://github.com/user-attachments/assets/84e142b6-7437-4268-9aef-423a496df393" />

##### 1.2.3 测试训练结果
<img width="1000" height="800" alt="6c7350312d602c0dd4291fc8c47ae3a7" src="https://github.com/user-attachments/assets/c0027923-1125-4891-b7fe-729d48d7ed5f" />

可以看到我这里训练的结果还是有1个出错的
#### 1.3 VGG16对CIFAR10分类
##### 1.3.1 定义 dataloader并训练
<img width="800" height="1000" alt="c02a62e396de1578814552a57aaf00c3" src="https://github.com/user-attachments/assets/af4851af-d72a-4f16-bb14-2067fefef5fd" />

##### 1.3.2 测试验证准确率
<img width="1200" height="600" alt="9b6fc28b02eafec2d90a74ec92113aee" src="https://github.com/user-attachments/assets/2034d654-c8fa-4b0d-af1b-d29c905a4bb6" />

### 2. 问题总结
#### 2.1 Adataloader 里面 shuffle 取不同值有什么区别？
shuffle=True会在每个epoch开始时打乱数据顺序；shuffle=False则保持原始顺序。主要是为了避免图片顺序在训练中影响结果
#### 2.2 transform 里，取了不同值，这个有什么区别？
transform中的RandomCrop可以随机裁取图片中的不同大小的区域，减少训练出现过拟合的现象；Normalize可以修改像素的映射关系，从而影响梯度传播速度和模型收敛稳定性；RandomHorizontalFlip可以设置图片旋转的概率，都是在提高训练时的随机性
#### 2.3 epoch 和 batch 的区别？
Epoch​​指整个数据集完整遍历一次的训练轮次；​​Batch​​是单次参数更新所用的数据子集。一个epoch包含多个batch迭代，batch越小噪声越多但内存需求低，epoch越多模型收敛越充分但可能过拟合
#### 2.4 1x1的卷积和 FC 有什么区别？主要起什么作⽤？
1x1卷积用于跨通道的信息融合或降维，保持空间维度；全连接层则展平所有维度，直接映射到输出类别。1x1卷积更高效，适合保留空间信息或调整通道数，FC通常仅用于网络末端分类
#### ​​2.5 residual leanring 为什么能够提升准确率？
ResNet 通过跳跃连接将输入直接加到深层输出上，解决了梯度消失和梯度爆炸的问题，使超深层网络可训练。其核心思想是让网络学习残差，而非直接拟合复杂映射
#### ​​2.6 代码练习二里，网络和1989年 Lecun 提出的 LeNet 有什么区别？
LeNet 仅有2个卷积层和3个FC层，代码练习二的网络比 LeNet 多了卷积层，使用了更小的卷积核、批量归一化和 ReLU 激活函数，并引入了数据增强，总体而言要比 LeNet 更深一些
#### ​​2.7 代码练习二里，卷积以后feature map 尺寸会变小，如何应用Residual Learning?
当卷积步长>1或池化导致特征图缩小时，可通过1x1卷积调整通道数，使跳跃连接的输入与输出维度匹配。例如，对H×W减半的情况，用步长2的1x1卷积同时降采样和调整通道
#### ​​2.8 有什么方法可以进一步提升准确率？
可以通过加深网络，调整学习率，使用更先进的数据增强或者是添加注意力机制等技术来提高准确率。

## 二、问题总结与体会
通过本次实验的代码练习和理论思考，我对卷积神经网络的基础概念有了更直观和深入的理解。在实验过程中通过调整参数我体会到了数据预处理中transform操作的重要性，不同的数据增强策略能有效提升模型的泛化能力，以及训练方式的选取对结果的影响是巨大的。本次实验让我意识到，调参和网络设计需要平衡模型复杂度与训练效率才能取得更好的效果。
