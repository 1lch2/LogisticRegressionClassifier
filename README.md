# Explanation
This is part of the assignment for study unit COMP5318 from University of Sydney.The goal of this assignment is to build a classifier to classify some grayscale images of the size 28x28 into a set of categories.<br />
Implemented with NumPy and h5py.
## Dataset
The dataset consists of a training set of 30,000 examples and a test set of 5,000 examples. They belong to 10 different categories. The validation set is not provided. The labels of the first 2,000 test examples are given. The rest 3,000 labels of the test set are reserved for marking purpose.<br />
Here are examples illustrating sample of the dataset (each class takes one row):<br />
<img src="https://github.com/1lch2/LogisticRegressionClassifier/blob/master/img/Dataset_image.jpg" alt="DataSet" title="DataSet" width="450" height="300" /><br />
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
## Code Instructions (for .ipynb only)
Put .ipynb file, 'Input' and 'Output' folder under the same directory.<br />
Training and testing data are located in 'Input' folder.<br />
predict_labels.h5 file will be generated in 'Output' folder.
## Performance
- Running time: 300s
- Accuracy: 84%
## Reference
[1]逻辑回归与交叉熵. https://zhuanlan.zhihu.com/p/38853901
