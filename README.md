# Explanation
This is part of the assignment for study unit COMP5318 and COMP5329 from University of Sydney.The goal of this assignment is to build a classifier to classify some grayscale images of the size 28x28 into a set of categories.<br />
Implemented with NumPy and h5py.
## Dataset 
### For LR and KNN
The dataset consists of a training set of 30,000 examples and a test set of 5,000 examples. They belong to 10 different categories. The validation set is not provided. The labels of the first 2,000 test examples are given. The rest 3,000 labels of the test set are reserved for marking purpose.<br />
Here are examples illustrating sample of the dataset (each class takes one row):<br />
<img src="https://github.com/1lch2/LogisticRegressionClassifier/blob/master/LogisticRegression/img/Dataset_image.jpg" alt="DataSet" title="DataSet" width="450" height="300" /><br />
There are 10 classes in total:<br />
0 T-shirt/Top<br />
1 Trouser<br />
2 Pullover<br />
3 Dress<br />
4 Coat<br />
5 Sandal<br />
6 Shirt<br />
7 Sneaker<br />
8 Bag<br />
9 Ankle boot <br />

### For NN
()
## Code Instructions (for .ipynb only)
Training and testing data are located in 'data' folder.<br />
predict_labels.h5 file will be generated in 'Output' folder.
## Performance
### Logistic Regression
- Running time: 300s
- Accuracy: 84%
### K-NN
- Running time: 929s
- Accuracy: 84.3%

## Reference
[1]逻辑回归与交叉熵. https://zhuanlan.zhihu.com/p/38853901 <br />
[2]12 回归算法 - 手写梯度下降代码. https://www.jianshu.com/p/a8aefacc4766 <br />
[3]机器学习之k-近邻（kNN）算法与Python实现. https://blog.csdn.net/moxigandashu/article/details/71169991 <br />
[4]严蔚敏.数据结构 (C语言版).清华大学出版社
