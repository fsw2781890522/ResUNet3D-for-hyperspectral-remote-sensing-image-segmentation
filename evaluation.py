import os
import time
import numpy as np
from skorch.callbacks import EpochScoring, BatchScoring
from skorch.callbacks import Callback
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, cohen_kappa_score, f1_score


def metric(neural_net, X, y):
    predict = neural_net.predict(X).argmax(1).flatten().tolist()
    y = y.flatten().tolist()
    accuracy = accuracy_score(y, predict)
    # precision = precision_score(y, predict, average='weighted')
    # recall = recall_score(y, predict, average='weighted')
    # kappa = cohen_kappa_score(y, predict)
    # f1 = f1_score(y, predict, average='weighted')

    # return accuracy, precision, recall, kappa, f1
    return accuracy


scoring = EpochScoring(
    metric,
    lower_is_better=False,
    name='metrics',
    on_train=True
)

train_scoring = EpochScoring(
    metric,
    lower_is_better=False,
    name='train_accuracy',
    on_train=True
)

valid_scoring = EpochScoring(
    metric,
    lower_is_better=False,
    name='valid_accuracy',
    on_train=False
)


# def metric(predict, label):
#     predict = predict.argmax(1)
#     # 生成混淆矩阵
#     conf_matrix = confusion_matrix(label, predict)
#     # print(conf_matrix)
#     # 计算总体精度
#     accuracy = accuracy_score(label, predict)
#     # 计算生产者精度（Precision）
#     precision = precision_score(label, predict, average='macro')
#     # 计算用户精度（Recall）
#     recall = recall_score(label, predict, average='macro')
#     # 计算Kappa系数
#     kappa = cohen_kappa_score(label, predict)
#     # 计算F1分数
#     f1 = f1_score(label, predict, average='macro')
#     # 计算IoU
#     # intersection = np.diag(conf_matrix).sum()
#     # union = np.sum(conf_matrix, axis=0) + np.sum(conf_matrix, axis=1) - intersection
#     # print(intersection)
#     # print(union)
#     # iou = float(intersection) / float(union)
#
#     return dict(
#         accuracy=accuracy,
#         precision=precision,
#         recall=recall,
#         kappa=kappa,
#         f1=f1
#         # iou=iou
#     )