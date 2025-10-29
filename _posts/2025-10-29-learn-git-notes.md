# 软件工程原理与实践实验报告

<center>姓名：李东旭  学号：23020007056</center>

| 姓名和学号         | 李东旭，23020007056                  |
| -------------------- | ------------------------------ |
| 课程 | 中国海洋大学25秋《软件工程原理与实践》 |
| 实验名称           | 实验4：MobileNet & ShuffleNet          |
## 一、实验内容
### 2. 代码作业
#### 2.1 定义 HybridSN 类

<img width="350" height="400" alt="495052d0ce012736595466457e2c1218" src="https://github.com/user-attachments/assets/f480a846-c668-474a-92d4-3343495b46e7" />
<img width="400" height="350" alt="b485a47faaab56381cf50102097ee98b" src="https://github.com/user-attachments/assets/f4266437-3281-40ee-80e1-0086ec1d69b3" />


#### 2.2 创建数据集

<img width="500" height="400" alt="1c69831f2424d53d5f1672e0617e06c7" src="https://github.com/user-attachments/assets/cd05aa0e-7ac8-4ad8-9fce-fa25a9bbb207" />


#### 2.3 开始训练

<img width="500" height="600" alt="2c138c8f975aa5d17651ea0fa4ee730b" src="https://github.com/user-attachments/assets/e235c6c0-4892-4c32-b9f5-c7df431df331" />


#### 2.4 模型测试

<img width="450" height="400" alt="9bb5a9f28534071cb72841b1564ff56e" src="https://github.com/user-attachments/assets/7eb2cc83-e297-4500-9b76-28b5599a3b07" />


#### 2.5 测试结果

<img width="450" height="500" alt="cb8238a617fceb66017a6d168c58e0b0" src="https://github.com/user-attachments/assets/41e8a574-3985-4f4a-9c30-7e47c47507b7" />


### 3. 思考题
#### 3.1 训练HybridSN，然后多测试几次，会发现每次分类的结果都不一样，请思考为什么？
训练结果不同的主要原因是神经网络训练中的随机性：权重初始化随机、数据打乱顺序随机、Dropout随机丢弃神经元、优化器内部随机因素。这些随机性导致每次训练收敛到不同的局部最优解。可通过设置随机种子或多次训练取平均来稳定结果。
#### 3.2 如果想要进一步提升高光谱图像的分类性能，可以如何改进？
可从四方面改进：数据层面采用先进的数据增强和类别平衡技术；模型架构引入注意力机制和更高效的网络模块；训练策略使用课程学习和自监督预训练；后处理加入空间平滑技术。还可探索Transformer等新架构的应用。
#### 3.3 depth-wise conv 和 分组卷积有什么区别与联系？
两者都是卷积变体以减少计算量。关键区别：分组卷积将通道分为多组处理，分组数小于通道数；Depth-wise是分组卷积的极端情况，分组数等于通道数，每个滤波器只处理一个输入通道。Depth-wise常与1×1卷积结合使用。
#### 3.4 SENet 的注意力是不是可以加在空间位置上？
可以。原始SENet关注通道注意力，但可扩展为空间注意力或两者结合。如CBAM模块同时包含通道和空间注意力，空间注意力通过通道维度池化和卷积学习空间权重，使模型关注图像关键区域。
#### 3.5 在 ShuffleNet 中，通道的 shuffle 如何用代码实现？
核心是通过reshape-transpose-reshape操作实现通道重排，促进组间信息交流，提升模型表现而不增加计算量。

## 二、问题总结与体会
通过本次高光谱图像分类实验，我对HybridSN网络结构及其应用有了更进一步的理解。在实验过程中，我认识到数据预处理中PCA降维和patch提取对模型性能的关键影响，合适的光谱维度和空间窗口大小能显著提升特征提取效果。训练过程中的随机性因素让我体会到设置随机种子对结果可复现性的重要性，同时也认识到Dropout等正则化技术虽然引入不确定性，却是防止过拟合的有效手段。
