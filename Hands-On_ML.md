## 机器学习基础

### 1. 机器学习纵览

#### 机器学习定义

一些ML的应用如spam filter和OCR（Optical Character Recognition）早在上世纪90年代起就已开始。2006年Geoffrey Hinton通过训练深度神经网络（deep neural network）识别手写数字更是实现了之前认为不可能完成的任务，自此深度学习（Deep Learning）进入人们视野。

本书认为ML是使用计算机通过数据进行学习的科学（和艺术），还给出了其他不同维度的定义（P4）。按：至于是从大量数据中学习（基于统计），还是少量数据加推演，进而只基于规则学习（如AlphaZero）目前都有人研究。

#### 为什么使用机器学习

原因是多方面的。有些问题无法抽象化进而转换为代码，如spam filter；有些则是数据的量级过大或没有已知算法，如语音识别以及数据挖掘（data mining）；还有则是像AlphaZero一样希望ML可以给人类一些从未有过启示。

#### 机器学习的类型

大体上可以从几个维度进行分类：

1.  是否需要人类监督（supervision)

    1.1 有监督学习，supervised learning。

    拥有一个包括结果的学习数据集。例如对分类问题（classfication）来说，一个标记（labeled）好的数据集；对回归问题（regression）来说，则是有期望值的数据集。

        算法：

        k-Nearest Neighbors
    
        Linear Regression
    
        Logistic Regression
    
        Support Vector Machines (SVMs)
    
        Decision Trees and Random Forests
    
        Neural networks2

    1.2 无监督学习，unsupervised learning

    1.3 半监督学习，semisupervised learning

    1.3 强化学习，reinforcement learning

2. 是否依据递增的数据持续学习：

    2.1 在线学些，online learning
    
    2.2 批量学习，batch learning

3. 基于数据比对或预测模型建立：

    3.1 基于实例学习，instance-based learning

    3.2 基于模型学习：model-based learning

一个ML应用可能是多种学习类型的组合及串联


