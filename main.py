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
from brats.utils import Trainer, iid, average_weights, compute_scores_per_classes, Trainer_central
from brats.criterion import BCEDiceLoss
from brats.configs.config import GlobalConfig, seed_everything

client = 4

def main():
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
    brats_dataset = get_dataloader(dataset=BraTS2020Dataset, path_to_csv='train_data.csv', phase='train', fold=0)
    trainer = Trainer_central(net=model,
                  dataset=BraTS2020Dataset,
                  criterion=BCEDiceLoss(),
                  lr=5e-4,
                  accumulation_steps=4,
                  batch_size=1,
                  fold=0,
                  num_epochs=10,
                  path_to_csv = config.path_to_csv,)
       
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

    trainer.run()
    
    
    val_dataloader = get_dataloader(BraTS2020Dataset, 'train_data.csv', phase='valid', fold=0)
    print(len(dataloader))
    model.load_state_dict(torch.load('./central.pth'))
    model.eval()
    dice_scores_per_classes, iou_scores_per_classes = compute_scores_per_classes(
        model, val_dataloader, ['WT', 'TC', 'ET']
    )
        
    dice_df = pd.DataFrame(dice_scores_per_classes)
    dice_df.columns = ['WT dice', 'TC dice', 'ET dice']
    iou_df = pd.DataFrame(iou_scores_per_classes)
    iou_df.columns = ['WT jaccard', 'TC jaccard', 'ET jaccard']
    val_metics_df = pd.concat([dice_df, iou_df], axis=1, sort=True)
    val_metics_df = val_metics_df.loc[:, ['WT dice', 'WT jaccard', 
                                        'TC dice', 'TC jaccard', 
                                        'ET dice', 'ET jaccard']]
    print(f"WT Dice score : {val_metics_df['WT dice'].mean()}")
    print(f"WT jaccard score : {val_metics_df['WT jaccard'].mean()}")
    print(f"TC Dice score : {val_metics_df['TC dice'].mean()}")
    print(f"TC jaccard score : {val_metics_df['TC jaccard'].mean()}")
    print(f"ET dice score : {val_metics_df['ET dice'].mean()}")
    print(f"ET jaccard score : {val_metics_df['ET jaccard'].mean()}")
    
    colors = ['#35FCFF', '#FF355A', '#96C503', '#C5035B', '#28B463', '#35FFAF']
    palette = sns.color_palette(colors, 6)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=val_metics_df.mean().index, y=val_metics_df.mean(), palette=palette, ax=ax)
    ax.set_xticklabels(val_metics_df.columns, fontsize=14, rotation=15)
    ax.set_title("Dice and Jaccard Coefficients from Validation", fontsize=20)

    for idx, p in enumerate(ax.patches):
        percentage = '{:.1f}%'.format(100 * val_metics_df.mean().values[idx])
        x = p.get_x() + p.get_width() / 2 - 0.15
        y = p.get_y() + p.get_height()
        ax.annotate(percentage, (x, y), fontsize=15, fontweight="bold")

    fig.savefig("result_central.png", format="png",  pad_inches=0.2, transparent=False, bbox_inches='tight')
    fig.savefig("result_central.svg", format="svg",  pad_inches=0.2, transparent=False, bbox_inches='tight')

if __name__ == '__main__':
    ## Train Process
    main()