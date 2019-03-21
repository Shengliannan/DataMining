
## 问题总结
### 一、ExecutableNotFound: failed to execute ['dot', '-Tsvg'], make sure the Graphviz executables are on your systems' PATH
```
graphviz是个软件，不能单独用Pip安装

首先，下载graphviz的安装包 ，地址：https://graphviz.gitlab.io/_pages/Download/Download_windows.html
```
>[参考](https://blog.csdn.net/c_daofeng/article/details/81077594)

dos下追加环境变量

set path=%path%;C:\Program Files (x86)\Graphviz2.38\bin;

dot -version 查看是否安装配置成功。



## sklearn入门&决策树在sklearn中的实现

> Python 3.7.1
>
> Scikit-learn 0.20.0
>
> Graphviz 0.8.4 没有画不出决策树, conda install python-graphviz | pip install graphviz
>
> Numpy 1.15.3,Pandas 0.23.4,Matplotlib 3.0.1,Scipy 1.1.0

1）如何找到最佳分枝

2）如何让决策树停止生长，防止过拟合。

slearn建模流程

1. 实例化，建立评估模型对象
2. 通过模型接口训练模型
3. 导入测试集，通过模型接口提取需要的信息

```
from sklearn import tree

clf = tree.DecisionTreeClassifer()
clf = clf.fit(x_train,y_train)
result = clf.score(X_test,y_test)
```

重要参数

criterion  用来决定不纯度的计算方法

同一棵树中，叶子节点的不纯度最低

1）entropy 信息熵（了解） 如何选择

2）gini 基尼系数（了解）

比起基尼系数，信息熵对不纯度更加敏感，对不纯度的惩罚最强。

