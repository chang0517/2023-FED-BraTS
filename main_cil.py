from tqdm import tqdm
import os
import time
import copy
from random import randint

import numpy as np
from scipy import stats
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import KFold

import nibabel as nib
import pydicom as pdm
import nilearn as nl
import nilearn.plotting as nlplt
import h5py

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as anim
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

import seaborn as sns
import imageio
from skimage.transform import resize
from skimage.util import montage

from IPython.display import Image as show_gif
from IPython.display import clear_output
from IPython.display import YouTubeVideo

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss


import albumentations as A
from albumentations import Compose, HorizontalFlip
from albumentations.pytorch import ToTensorV2 

import warnings
warnings.simplefilter("ignore")
from brats.model import UNet3d
from brats.data.brats import get_dataloader, get_dataset
from brats.data.brats import BraTS2020Dataset, DatasetSplit
from brats.utils import ADDTrainer, iid, average_weights, compute_scores_per_classes
from brats.criterion import BCEDiceLoss
from brats.configs.config import GlobalConfig, seed_everything
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
client = 4

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # torch.cuda.set_device(d모evice)
    print('Count of using GPUs:', torch.cuda.device_count())
    config = GlobalConfig()
    seed_everything(config.seed)

    dataloader = get_dataloader(dataset=BraTS2020Dataset, path_to_csv='train_data.csv', phase='valid', fold=0)
    print(len(dataloader))

    data = next(iter(dataloader))
    print(data['Id'], data['image'].shape, data['mask'].shape)

    img_tensor = data['image'].squeeze()[0].cpu().detach().numpy() 
    mask_tensor = data['mask'].squeeze()[0].squeeze().cpu().detach().numpy()
    print("Num uniq Image values :", len(np.unique(img_tensor, return_counts=True)[0]))
    print("Min/Max Image values:", img_tensor.min(), img_tensor.max())
    print("Num uniq Mask values:", np.unique(mask_tensor, return_counts=True))

    image = np.rot90(montage(img_tensor))
    mask = np.rot90(montage(mask_tensor)) 

    fig, ax = plt.subplots(1, 1, figsize = (20, 20))
    ax.imshow(image, cmap ='bone')
    ax.imshow(np.ma.masked_where(mask == False, mask),
               cmap='cool', alpha=0.6)
    model = UNet3d(in_channels=4, n_classes=3, n_channels=24).to('cuda')
    global_model = copy.deepcopy(model)
    brats_dataset = get_dataset(dataset=BraTS2020Dataset, path_to_csv='train_data.csv', phase='train', fold=0)
    dict_users = iid(brats_dataset, client)

    trainers = []
    for c in range(client):
        trainer = Trainer(net=model,
                        dataset=DatasetSplit(brats_dataset, dict_users[c]),
                        criterion=BCEDiceLoss(),
                        lr=5e-4,
                        accumulation_steps=4,
                        batch_size=1,
                        fold=0,
                        num_epochs=1,
                        path_to_csv = config.path_to_csv)
        trainers.append(trainer)
    
    if config.pretrained_model_path is not None:
        trainer.load_predtrain_model(config.pretrained_model_path)

        # if need - load the logs.      
        train_logs = pd.read_csv(config.train_logs_path)
        trainer.losses["train"] =  train_logs.loc[:, "train_loss"].to_list()
        trainer.losses["val"] =  train_logs.loc[:, "val_loss"].to_list()
        trainer.dice_scores["train"] = train_logs.loc[:, "train_dice"].to_list()
        trainer.dice_scores["val"] = train_logs.loc[:, "val_dice"].to_list()
        trainer.jaccard_scores["train"] = train_logs.loc[:, "train_jaccard"].to_list()
        trainer.jaccard_scores["val"] = train_logs.loc[:, "val_jaccard"].to_list()
    
    for c in range(client):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        #trainers[c].net = torch.nn.DataParallel(trainers[c].net, [0, 1])
        # trainers[c].net.to(device)
        #trainers[c].discriminator = torch.nn.DataParallel(trainers[c].discriminator, [0, 1])
        # trainers[c].discriminator.to(device)
        #trainers[c].global_net = torch.nn.DataParallel(trainers[c].global_net, [0, 1])
        # trainers[c].global_net.to(device)
     
    
    #global_para = torch.nn.DataParallel(global_model).state_dict()
    
    for round in range(200):
        tmp = []
        for c in range(client):
            trainers[ ].run()
            tmp.append(trainers[c].get_model_para())
        global_para = average_weights(tmp)
        global_model.load_state_dict(global_para)
        torch.save(global_para, f'./model_save/fedadd_ill_{round}.pt')
        
if __name__ == '__main__':
    ## Train Process
    main()