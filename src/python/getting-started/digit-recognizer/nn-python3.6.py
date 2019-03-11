#!/usr/bin/python
# coding: utf-8
'''
Created on 2018-05-14
Update  on 2018-05-14
Author: 平淡的天
Github: https://github.com/apachecn/kaggle
'''

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
import os

datasets_path = "C:/Users/ezjinfe/datasets"

data_dir = '{}/getting-started/digit-recognizer/'.format(datasets_path)
output_path = os.path.join(data_dir, 'output')
if not os.path.exists(output_path):
    os.makedirs(output_path)


train_data = pd.read_csv(r"{}\input\train.csv".format(data_dir))
test_data = pd.read_csv(r"{}\input\test.csv".format(data_dir))

data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)
data.drop(['label'], axis=1, inplace=True)
label = train_data.label

pca = PCA(n_components=100, random_state=34)
data_pca = pca.fit_transform(data)

Xtrain, Ytrain, xtest, ytest = train_test_split(
    data_pca[0:len(train_data)], label, test_size=0.1, random_state=34)

clf = MLPClassifier(
    hidden_layer_sizes=(100, ),
    activation='relu',
    alpha=0.0001,
    learning_rate='constant',
    learning_rate_init=0.001,
    max_iter=200,
    shuffle=True,
    random_state=34)

clf.fit(Xtrain, xtest)
y_predict = clf.predict(Ytrain)

zeroLable = ytest - y_predict
rightCount = 0
for i in range(len(zeroLable)):
    if list(zeroLable)[i] == 0:
        rightCount += 1
print('the right rate is:', float(rightCount) / len(zeroLable))

result = clf.predict(data_pca[len(train_data):])

i = 0


def saveResult(result, csvName):
    i = 0
    n = len(result)
    print('the size of test set is {}'.format(n))
    with open(os.path.join(output_path, "{}.csv".format(csvName)), 'w') as fw:
        fw.write('{},{}\n'.format('ImageId', 'Label'))
        for i in range(1, n + 1):
            fw.write('{},{}\n'.format(i, result[i - 1]))
    print('Result saved successfully... and the path = {}'.format(csvName))

saveResult(result, "result_nn")