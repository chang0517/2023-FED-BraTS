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
from brats.utils import Trainer, iid, average_weights, compute_scores_per_classes
from brats.criterion import BCEDiceLoss
from brats.configs.config import GlobalConfig, seed_everything
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def main():
    config = GlobalConfig()
    seed_everything(config.seed)
    
    
    val_dataloader = get_dataloader(BraTS2020Dataset, 'train_data.csv', phase='valid', fold=0)
    print(len(val_dataloader))
    # model_name = 'last_epoch_model.pth'
    best_score = 0
    
    for i in tqdm(range(50, 0, -1)):
        add_model_name = f'fedadd_{i}'
        avg_model_name = f'fedavg_{i}'
        add_valid_model = UNet3d(in_channels=4, n_classes=3, n_channels=24).to('cuda')
        avg_valid_model = UNet3d(in_channels=4, n_classes=3, n_channels=24).to('cuda')
        
        add_valid_model.load_state_dict(torch.load(f'./model_save/{add_model_name}.pt',map_location = 'cuda:0'))
        add_valid_model.eval()
        avg_valid_model.load_state_dict(torch.load(f'./model_save/{avg_model_name}.pt',map_location = 'cuda:0'))
        avg_valid_model.eval()
        
        
        add_dice_scores_per_classes, _ = compute_scores_per_classes(
        add_valid_model, val_dataloader, ['WT', 'TC', 'ET']
        )   
        avg_dice_scores_per_classes, _ = compute_scores_per_classes(
        avg_valid_model, val_dataloader, ['WT', 'TC', 'ET']
        )  
        mean_add = sum(add_dice_scores_per_classes['WT'])/len(add_dice_scores_per_classes['WT'])
        mean_avg = sum(add_dice_scores_per_classes['WT'])/len(add_dice_scores_per_classes['WT'])
        if (mean_add > mean_avg) & (best_score < mean_add):
            best_score = mean_add
            torch.save(add_valid_model.state_dict(),f'./add_avg/{add_model_name}.pt')
    
    # validation score\
    # best_model_name = 'fedavg_108'
    # valid_model.load_state_dict(torch.load(f'./model_save/{best_model_name}.pt',map_location = 'cuda:0'))
    # dice_scores_per_classes, iou_scores_per_classes = compute_scores_per_classes(
    #     valid_model, val_dataloader, ['WT', 'TC', 'ET']
    # )
    # dice_df = pd.DataFrame(dice_scores_per_classes)
    # dice_df.columns = ['WT dice', 'TC dice', 'ET dice']
    # iou_df = pd.DataFrame(iou_scores_per_classes)
    # iou_df.columns = ['WT jaccard', 'TC jaccard', 'ET jaccard']
    # val_metics_df = pd.concat([dice_df, iou_df], axis=1, sort=True)
    # val_metics_df = val_metics_df.loc[:, ['WT dice', 'WT jaccard', 
    #                                     'TC dice', 'TC jaccard', 
    #                                     'ET dice', 'ET jaccard']]
    # print(f"WT Dice score : {val_metics_df['WT dice'].mean()}")
    # print(f"WT jaccard score : {val_metics_df['WT jaccard'].mean()}")
    # print(f"TC Dice score : {val_metics_df['TC dice'].mean()}")
    # print(f"TC jaccard score : {val_metics_df['TC jaccard'].mean()}")
    # print(f"ET dice score : {val_metics_df['ET dice'].mean()}")
    # print(f"ET jaccard score : {val_metics_df['ET jaccard'].mean()}")
    
    # # visualization
    # colors = ['#35FCFF', '#FF355A', '#96C503', '#C5035B', '#28B463', '#35FFAF']
    # palette = sns.color_palette(colors, 6)

    # fig, ax = plt.subplots(figsize=(12, 6))
    # sns.barplot(x=val_metics_df.mean().index, y=val_metics_df.mean(), palette=palette, ax=ax)
    # ax.set_xticklabels(val_metics_df.columns, fontsize=14, rotation=15)
    # ax.set_title("Dice and Jaccard Coefficients from Validation", fontsize=20)

    # for idx, p in enumerate(ax.patches):
    #     percentage = '{:.1f}%'.format(100 * val_metics_df.mean().values[idx])
    #     x = p.get_x() + p.get_width() / 2 - 0.15
    #     y = p.get_y() + p.get_height()
    #     ax.annotate(percentage, (x, y), fontsize=15, fontweight="bold")

    # fig.savefig(f"result_best_{best_model_name}.png", format="png",  pad_inches=0.2, transparent=False, bbox_inches='tight')
    # fig.savefig(f"result_best_{best_model_name}.svg", format="svg",  pad_inches=0.2, transparent=False, bbox_inches='tight')
if __name__ == '__main__':
    ## Train Process
    main()