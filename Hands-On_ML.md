## 机器学习基础

### 1. 机器学习纵览

#### 机器学习定义

一些 ML 的应用如 spam filter 和 OCR（Optical Character Recognition）早在上世纪 90 年代起就已开始。2006 年 Geoffrey Hinton 通过训练深度神经网络（deep neural network）识别手写数字更是实现了之前认为不可能完成的任务，自此深度学习（Deep Learning）进入人们视野。

本书认为 ML 是使用计算机通过数据进行学习的科学（和艺术），还给出了其他不同维度的定义（P4）。按：至于是从大量数据中学习（基于统计），还是少量数据加推演，进而只基于规则学习（如 AlphaZero）目前都有人研究。

#### 为什么使用机器学习

原因是多方面的。有些问题无法抽象化进而转换为代码，如 spam filter；有些则是数据的量级过大或没有已知算法，如语音识别以及数据挖掘（data mining）；还有则是像 AlphaZero 一样希望 ML 可以给人类一些从未有过启示。

#### 机器学习的类型

大体上可以从几个维度进行分类：

1.  是否需要人类监督（supervision)

    1.1 有监督学习，supervised learning。

    拥有一个包括期望值的学习数据集。例如对分类问题（classfication）来说，一个标记（labeled）好的数据集；对回归问题（regression）来说，则是有期望值的数据集。

    算法：

    - k-Nearest Neighbors (KNN)
    - Linear Regression
    - Logistic Regression
    - Support Vector Machines (SVMs)
    - Decision Trees and Random Forests
    - Neural networks2

      1.2 无监督学习，unsupervised learning

    用于学习的数据集没有期望值，需要 ML 自己学习并给出期望值。

    算法：

    聚类算法（Clustering）：

    - k-Means
    - Hierarchical Cluster Analysis (HCA)
    - Expectation Maximization

    可视化及降维（Visualization and dimensionality reduction）

    - Principal Component Analysis (PCA)
    - Kernel PCA
    - Locally-Linear Embedding (LLE)
    - t-distributed Stochastic Neighbor Embedding (t-SNE)

    关联规则学习（Association rule learning）

    - Apriori
    - Eclat

      1.3 半监督学习，semisupervised learning

    用于学习的数据集部分有期望值，或经过初步非监督学习（如 clustering）可以辅助产生用于监督学习的期望值（如 label）。这类学习通常为监督学习和非监督学习的结合。

    1.4 强化学习，reinforcement learning

    强化学习中的代理（agent）通过观察环境（environment）后采取行动（action），根据结果（reward/penality）调整策略（policy）。\*\*\*\*

2.  是否依据递增的数据持续学习：

    2.1 在线学习，online learning

    2.2 批量学习，batch learning

    基于诸如计算量过大等原因无法持续学习时采用线下学习（offline learning）

3.  基于数据比对或预测模型建立：

    3.1 基于实例学习，instance-based learning

    3.2 基于模型学习：model-based learning

    - ϴ（theta）常被用于表示模型系数
    - utility function / fitness function，指示模型有多好
    - cost function, 指示模型有多糟糕

一个 ML 应用可能是多种学习类型的组合及串联

#### 机器学习的主要挑战

导致学习不成功的原因主要来自两方面，坏数据和坏算法。

1. 坏数据

   - 数据量不足
   - 数据代表性不足，会造成取样偏差（Sampling bias），或取样噪声（Sampling noise）
   - 数据质量不足
   - 无关信息，需要 Feature engineering 优化，如特征选取，提取，创建等。

2. 坏算法

   - 过拟合（Overfitting）
     - 通过正则化（Regularization）减轻过拟。
     - 通过超参数（Hyperparameter）调节正则化程度
   - 欠拟合（Underfitting）
     - 通常由于模型过于简单或限制过多导致

#### 测试和验证

- 拆分数据集为训练集（Training set）和测试集（Test set）一般使用二八原则。
- 泛化误差（Generalization error）为预测样本集外数据的误差，又叫样本外误差（Out-of-sample error）。
- 验证集（Validation set）常被用于调试超参。为了节约可以将数据集拆分为两组互为训练集和验证集。
- 没有免费的午餐（No Free Lunch， NFL），除非测试无法确定一种模型一定优于另一种。

### 2. 从头至尾做一个 ML 项目

#### 数据源

P34 列出了部分有用的数据源。本章使用 StatLib repository 的加州房价数据集，预测湾区的房价中位数。

#### 整体分析

1. 框定问题范围

   了解问题上下文并分析该 ML 项目应属的类型，监督，非监督；分类，回归；批量，在线等。

   本章的项目是一个监督型回归问题的 ML 项目，相对于之前根据 GDP 预测幸福指数的问题（单变量回归，Univariate regression）， 这是一个多变量回归问题（Multivariate regression）

   - 信号（Signal），指供 ML 系统学习的一组信息
   - 管道（Pipeline），由多个组件连接而成的数据处理过程

2. 选择评估方式

   - 均方根误差，Root Mean Square Error (RMSE)

     是指预测误差集的标准偏差（Standard Deviation），或可以说是预测值集和实际值集的差异程度。

     - 均方误差（Mean Square Error）

     - 标准差（Standard Deviation），用 σ（sigma）代表，是方差（Variance）的平方根。

     描述一组数据中值的分布情况，当数据正态分布（Normal distribution/Gaussian distribution）时约有 68%的值与平均值的差在 1σ 以内，约 98%在 2σ 以内，99.7%在 3σ 以内。

     - 方差（Variance），反应数据集离散程度。数据与均值差方的平均值。

   - 平均绝对误差，Mean Absolute Error (MAE)/Average Absolute Deviation

   - 范数，norms。意为范本，基准

     衡量向量距离时使用范数，范数分幂级，记作||x||i 或 ℓi

     - 0 级为向量基数（元素数量）
     - 1 级为元素绝对值之和，又称曼哈顿范数，与 MAE 相关
     - 2 级为欧式范数，与 RSME 相关
     - +inf 级为最大绝对值， -inf 级为最小绝对值

范数级数越高对大数（远距离）也就是 outlier 约敏感，所以 RMSE（2 级）比 MAE（1 级）对 outlier 更敏感

3. 创建测试集

- 数据监听偏差（Data snooping bias），再择选模型前拆分
- 取样偏差（Sampling bias），通过分层取样（Stratified sampling）解决
- 拆分数据可使用 Scikit-Learn