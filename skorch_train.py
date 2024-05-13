import os
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.optim import optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LinearLR
from torchvision import transforms
from skorch import NeuralNet, NeuralNetClassifier
from skorch.callbacks import EpochScoring, BatchScoring
from skorch.callbacks import Callback
from skorch.callbacks import LRScheduler
from skorch.dataset import Dataset, ValidSplit
from skorch.helper import predefined_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, cohen_kappa_score, f1_score

# self library
from model3D import ResUNet3D
from dataset import MyDataset, calc_dataset_stats
from evaluation import train_scoring, valid_scoring

if __name__ == '__main__':
    # 定义训练设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 定义参数
    num_classes = 4 + 1  # unclassified need to be considered
    num_bands = 150
    epochs = 50
    batch_size = 8
    root_dir = r'H:\thesis\Jabal_Damkh'

    # total_data = MyDataset(raster_dir=os.path.join(root_dir, r'dataset\total'))
    train_data = MyDataset(raster_dir=os.path.join(root_dir, r'dataset\train'))
    valid_data = MyDataset(raster_dir=os.path.join(root_dir, r'dataset\valid'))
    train_mean, train_std = calc_dataset_stats(train_data)
    valid_mean, valid_std = calc_dataset_stats(valid_data)
    # print(mean.shape, std.shape)

    train_data = MyDataset(raster_dir=os.path.join(root_dir, r'dataset\train'),
                           image_transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(train_mean, train_std)
                           ]),
                           label_transform=transforms.ToTensor()
                           )
    valid_data = MyDataset(raster_dir=os.path.join(root_dir, r'dataset\valid'),
                           image_transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(valid_mean, valid_std)
                           ]),
                           label_transform=transforms.ToTensor()
                           )
    # total_data = MyDataset(raster_dir=os.path.join(root_dir, r'dataset\total'),
    #                        image_transform=transforms.Compose([
    #                            transforms.ToTensor(),
    #                            transforms.Normalize(mean, std)
    #                        ]),
    #                        label_transform=transforms.ToTensor()
    #                        )

    # 创建网络模型
    neural_network = NeuralNet(
        module=ResUNet3D,
        module__num_bands=num_bands,
        module__num_classes=num_classes,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.SGD,
        optimizer__lr=0.01,
        max_epochs=epochs,
        batch_size=batch_size,
        iterator_train__shuffle=True,
        device=device,
        train_split=predefined_split(valid_data),  # By default, 20% of the incoming data is reserved for validation
        callbacks=[train_scoring, valid_scoring, ('lr_scheduler',
                                                  LRScheduler(policy=StepLR,
                                                              step_size=5,
                                                              gamma=0.8))]
    )

    # 训练模型
    neural_network.fit(train_data)

    print('Saving model...')
    model_fn = os.path.join(root_dir, 'ResUnet3D.pkl')
    if os.path.exists(model_fn):
        os.remove(model_fn)
    with open(model_fn, 'wb') as f:
        pickle.dump(neural_network, f)

    # accuracy = accuracy_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred, average='micro')
    #
    # print("Accuracy:", accuracy)
    # print("Precision:", precision)

    # writer.close()

    #
