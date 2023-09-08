import numpy as np
import torch
import os

class GlobalConfig:
    # root_dir = './data/brats20-dataset-training-validation'
    # root_dir = '../data/BraTS2020'
    root_dir = '../input/brats20-dataset-training-validation'
    train_root_dir = '../input/BraTS2020/MICCAI_BraTS2020_TrainingData'
    test_root_dir = '../input/BraTS2020/MICCAI_BraTS2020_ValidationData'
    path_to_csv = './train_data.csv'
    pretrained_model_path = './brats20logs/unet/last_epoch_model.pth'
    train_logs_path = './brats20logs/unet/train_log.csv'
    ae_pretrained_model_path = './brats20logs/ae/autoencoder_best_model.pth'
    tab_data = './brats20logs/data/df_with_voxel_stats_and_latent_features.csv'
    seed = 55
    
def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
config = GlobalConfig()
seed_everything(config.seed)
