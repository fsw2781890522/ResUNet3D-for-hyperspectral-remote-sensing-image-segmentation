import os
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import spectral.io.envi as e
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from skorch import NeuralNet, NeuralNetClassifier
from skorch.callbacks import EpochScoring, BatchScoring
from skorch.callbacks import Callback
from skorch.dataset import Dataset, ValidSplit
from skorch.helper import predefined_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, cohen_kappa_score, f1_score

# self library
from model3D import ResUNet3D
from dataset import MyDataset, calc_dataset_stats, reconstruct_image
from evaluation import train_scoring, valid_scoring

if __name__ == '__main__':
    root_dir = r'H:\thesis\Jabal_Damkh'

    original_raster = e.open(r"H:\thesis\Jabal_Damkh\Jabal_Damkh_partial_labelraster_clip_mod.hdr")
    nl, ns, nb = original_raster.shape
    new_meta = original_raster.metadata.copy()

    total_data = MyDataset(raster_dir=os.path.join(root_dir, r'dataset\total'))
    mean, std = calc_dataset_stats(total_data)

    total_data = MyDataset(raster_dir=os.path.join(root_dir, r'dataset\total'),
                           image_transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(mean, std)
                           ]),
                           label_transform=transforms.ToTensor()
                           )

    print('Loading trained model...')
    with open(os.path.join(root_dir, 'ResUnet3D.pkl'), 'rb') as f:
        neural_network = pickle.load(f)

    print('Predicting...')
    predict = neural_network.predict(total_data).argmax(1)  # shape: (n_patches, h_patch, w_patch)
    predict = reconstruct_image(predict, nl, ns, patch_size=(16, 16))

    print('Saving...')
    # Delete spectra-related keys
    spec_keys = ['wavelength', 'fwhm', 'wavelength units', 'data gain values', 'data offset values', 'bbl']
    for spec_key in spec_keys:
        if spec_key in new_meta:
            new_meta.pop(spec_key)
    for key in new_meta.keys():
        value = new_meta[key]
        if isinstance(value, list) and len(value) == nb:
            new_meta[key] = value[0]
        else:
            new_meta[key] = value

    predict_fn = r"H:\thesis\Jabal_Damkh\Jabal_Damkh_partial_predict.hdr"
    if os.path.exists(predict_fn):
        os.remove(predict_fn)
        os.remove(predict_fn.replace('.hdr', '.img'))
    e.save_classification(predict_fn, predict, metadata=new_meta)
