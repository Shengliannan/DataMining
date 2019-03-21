# DataMining


GBDT(Gradient Boosting Decision Tree)又叫(Multiple Additive Regression Tree)，是一种迭代的决策树算法，该算法由多棵决策树组成，所有树的结论累加起来做最终答案。它在被提出之初就和SVM一起被认为是泛化能力(generalization)较强的算法。近些年更因为被用于搜索排序的机器学习模型而引起大家的关注。

## 数据处理

数据标准化（数据缩放）

- 标准化

  标准化是将数据按比例缩放，使之落入一个小的特征区间。

- 规格化（归一化）

  归一化是一种***简化计算***的方式，即将有量纲的表达式，经过变换，化为无量纲的表达式，成为纯量，把数据映射到0~1范围之内处理。

  > 思考：
  >
  > 标准化和规格化的区别是什么？各自的应用场景？

## 特征工程（特征选取）

计算特征的信息量

## 算法开发

## 工具
### IPython

魔法(Magic)命令







## 数据处理库

###  Numpy

> [Numpy教程](http://www.runoob.com/numpy/numpy-tutorial.html)



### Pandas
>[Pandas教程](https://www.yiibai.com/pandas)


Pandas是***基于Numpy***的一个开源Python库，它被广泛用于快速分析数据，以及数据清洗和准备等工作。它的名字来源是由“Panel Data”(面板数据，一个计量经济学名词)两个单词拼成的。***简单的说，你可以把Pandas看作是Python版的Excel。***

一、安装

```
# Anaconda安装
conda install pandas
# pip安装
pip install pandas
```

二、Pandas数据结构

- 一维数组

  Series

Series是一种以为数组，和Numpy里的数组很相似。事实上，Series基本上就是基于Numpy的数组对象来的。和NumPy的数组不同，Series能为数据自定义标签，也就是索引(index)，然后通过索引来访问数组中的数据。

创建一个Series的基本语法如下：

```python
# data参数可以是任意数据对象，比如字典、列表甚至是NumPy数组，而index参数则是对data的索引值，类似字典的key。
my_series = pd.Series(data,index)
```

eg：创建一个Series对象，并用字符串对数字列表进行索引：

```python
import pandas as pd
countries = ['USA','Nigeria','France','Ghana']
my_data = [100,200,300,400]
my_series = pd.Series(my_data,countries)
print(my_series)
print(my_series['USA']) # output:100
```

- 二维数组

  DataFrames
Pandas的DataFrame（数据表）是一种2维数据结构，数据以表格的形式存储，分成若干行和列。
常见操作：选取、替换行或列数据，重组数据表，修改索引，多重筛选

  

###  Scipy
###  matplotlib

## TensorFlow

## TensorFlow与Scikit-learn的区别

## Scikit-learn 

#### 一、Intro

> [scikit官网](https://scikit-learn.org/stable/)
>
> [莫烦PYTHON](https://morvanzhou.github.io/)

- Simple and efficient tools for data mining and data analysis
- Accessible to everybody, and reusable in various contexts
- Built on NumPy, SciPy, and matplotlib
- Open source, commercially usable - BSD license

scikit-learn(简记sklearn)，是用python实现的机器学习算法库。sklearn可以实现数据预处理、分类、回归、降维、模型选择等常用的机器学习算法。sklearn是基于Numpy，Scipy，matplotlib的。
- Numpy python 实现的开源科学计算包。它可以定义高维数组对象；矩阵计算和随机数生成等函数。
- Scipy python实现的高级科学计算包。它和Numpy联系很密切，Scipy一般都是操控Numpy数据来进行科学计算，所以可以说是基于Numpy。Scipy有很多子模块可以应对不同的应用，例如插值运算，优化算法、图像处理、数字统计等。
- matplotlib python实现的作图包。使用matplotlib能够非常简单的可视化数据，仅需要几行代码，便可以生成直方图、功率图、条形图、错误图、散点图等。

#### 二、小项目开发
1. 安装numpy
    ```
    pip3 install numpy
    ```
2. 安装scipy
    ```
    pip3 install scipy
    ```
3. 安装scikit-learn
    ```
    pip3 install -U scikit-learn
    ```
4. 安装jupyter
    ```
    python -m pip install jupyter
    ```
5. 启动jupyter
    ```
    jupyter notebook
    ```
6. 编写代码

   ```python
   #预测iris
   
   #!/usr/bin/env python
   # coding: utf-8
   
   from sklearn import datasets
   import numpy as np
   from sklearn.model_selection import train_test_split
   from sklearn.neighbors import KNeighborsClassifier
   iris=datasets.load_iris()
   iris_X=iris.data
   iris_y=iris.target
   print(iris_X[:2,:])
   print(iris_y)
   # 测试集占总数据集的大小为30%
   X_train,X_test,y_train,y_test=train_test_split(iris_X,iris_y,test_size=0.3)
   print(y_train)
   knn=KNeighborsClassifier()
   knn.fit(X_train,y_train)
   print(knn.predict(X_test))
   print(y_test)
   ```


注意：调包侠

#### 三、sklearn问题总结

1.sklearn.cross_validation 0.18版本废弃警告及解决方法，cross_validation会出现横线

解决办法：
改为从 ***sklearn.model_selection*** 中调用***train_test_split***函数可以解决此问题。

eg：

```python
# 数据分割
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=33)
```



### 经典分类回归算法

#### 线性回归

#### 逻辑回归

#### 岭回归

- 过拟合，正则项
- 

### 聚类算法和特征选择

### 模型选择，调参和数据预处理

### 文本分类

### TFIDF

### fasttext

### w2v

### xgb

### gbdt+lr

### 深度学习

### 交叉熵损失

### CNN适用场景

### boosting和bagging的区别

### 决策树如何做特征选择

### 旅行商问题

### xgboost和gdbt和随机森林的区别

### sql leftjoin 和 rightjoin的区别

### 损失函数（Hinge loss）



## 特征工程

AUC的意义

1. auc只反映模型对正负样本排序能力强弱，对score的大小和精度没有要求；
2. auc越高模型的排序能力越强，理论上，当模型把所有正样本排在负样本之前时，auc为1.0，是理论最大值。



人工智能非常依赖数据量和硬件资源。

## 机器学习及深度学习的当前应用

- 语音助手
- 智能客服
- 推荐系统
- 信用卡防欺诈
- 过滤垃圾邮件
- 疾病监测与诊断

