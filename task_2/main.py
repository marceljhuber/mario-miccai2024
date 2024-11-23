# encoding: utf-8

"""
Main file for training a model for Task2.
"""

from model import LinearProjection
from trainer import *

from dataset import NinetyDaysLatentMatching

from torch_device import get_device

from torch.utils.data import DataLoader
from torch import nn
import pandas as pd
import torch

import datetime
import os

# Define Hyperparameters
hp = {
    'img_size': (224, 224),
    'image_path': os.path.join('..', 'data_1', 'data_task1', 'train'),
    'tensor_path': os.path.join('..', 'data_1', 'data_task1_fe', 'train'),
    'train_csv_path': os.path.join('..', 'data_1', '90_days_data.csv'),
    'valid_csv_path': None,  # os.path.join('..', 'data_1', 'train_splits', 'valid'),
    'feature_extractor_oct_path': None,  # os.path.join('..', 'ckpts', 'RetFoundFeatureExtractor', 'RETFound_oct_weights.pth'),
    'feature_extractor_fundus_path': None,  # os.path.join('..', 'ckpts', 'RetFoundFeatureExtractor', 'RETFound_cfp_weights.pth'),
    'save_path': os.path.join(f'..', 'ckpts', '90DaysLatentMatching', f'{datetime.datetime.now().strftime("%d_%m_%Y-%H_%M_%S")}'),
    'save_every': 1,
    'batch_size': 1,
    'n_workers': 0,
    'n_epochs': 100 + 1,
    'dropout_p': 0.0,
    'learning_rate': 0.001,
    'device': get_device(),
    'in_features': 1024,
    'out_features': 1024,
    'hidden_features': 1024,
    'return_csv': True,

    'dataset': NinetyDaysLatentMatching,
    'trainer_class': TwoVisitOCTTrainer,
    'model': LinearProjection,
}

scheduler_hp = {
    'factor': 0.9,
    'patience': 4,
    'threshold': 1e-4,
    'cooldown': 0,
    'min_lr': 1e-10,
}

# tracker = wandb.init(
#     project="MICCAI Challenge Task 1 - Two-Visit OCT",
#     name="90DaysLatentMatching",
#     config=hp,
# )

os.makedirs((hp['save_path']), exist_ok=True)

data = pd.read_csv(hp['train_csv_path'])
train_valid_th = 0.8
train_samples = int(len(data) * train_valid_th)

train_data = hp['dataset'](
    data=data[:train_samples],
    tensor_path=hp['tensor_path'],
    device=hp['device'] if hp['device'] == torch.device('cuda') else torch.device('cpu'),
)

train_loader = DataLoader(
    train_data,
    batch_size=hp['batch_size'],
    num_workers=hp['n_workers'],
    shuffle=True,
)

valid_data = hp['dataset'](
    data=data[train_samples:],
    tensor_path=hp['tensor_path'],
    device=hp['device'] if hp['device'] == torch.device('cuda') else torch.device('cpu'),
)

valid_loader = DataLoader(
    valid_data,
    batch_size=hp['batch_size'],
    num_workers=hp['n_workers'],
    shuffle=False,
)

# ### ### ###

model = hp['model'](
    in_features=hp['in_features'],
    hidden_features=hp['hidden_features'],
    out_features=hp['out_features'],
    bias=True,
    dropout_p=hp['dropout_p'],
).to(hp['device'])

# tracker.watch(
#     model,
#     log="all",
# )


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    numerator = torch.dot(x, y)
    denominator = torch.linalg.norm(x) * torch.linalg.norm(y)
    return numerator / denominator


criterion = nn.CosineSimilarity(dim=1)  # cosine_similarity
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=hp['learning_rate'],
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    **scheduler_hp,
)

trainer = hp['trainer_class'](
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    feature_extractor_oct=None,  # feature_extractor_oct,
    feature_extractor_fun=None,  # feature_extractor_fun,
    scheduler=scheduler,
    tracker=None,  # tracker,
    train_loader=train_loader,
    valid_loader=valid_loader,
)

trainer(
    num_epochs=hp['n_epochs'],
    save_path=hp['save_path'],
    save_every=hp['save_every'],
)

# wandb.finish()
