import matplotlib.pyplot as plt
import numpy as np
import os
import gc
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, cohen_kappa_score, accuracy_score
from sklearn.model_selection import KFold
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from typing import List, Tuple

import utils
import wandb

from Models import *
from Transforms import *
from make_split import stratify_patients

# Constants
CHECKPOINTS_DIR = 'checkpoints'
DATA_DIR = os.path.join('..', 'data_1')
TRAIN_CSV = os.path.join(DATA_DIR, 'df_task1_train_challenge.csv')
VAL_CSV = os.path.join(DATA_DIR, 'df_task1_train_challenge.csv')
IMG_PREFIX = os.path.join(DATA_DIR, 'data_task1', 'train')

# Create checkpoints directory if it doesn't exist
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# Set random seeds for reproducibility
utils.set_random_seeds(42)

# Hyperparameters
hp = {
    'num_epochs': 10,
    'learning_rate': 1e-5,
    'weight_decay': 1e-4,
    'batch_size': 20,
    'model': CustomDinoV1,
    'transform_train': transformV21,
    'transform_valid': transformV21,
    'class_weights': None,
    'num_workers': 0,
    'csv_train': TRAIN_CSV,
    'csv_valid': VAL_CSV,
    'img_prefix': IMG_PREFIX,
    'min_delta': 0.05,
    'patience': 2,
}


class CustomDataset(Dataset):
    def __init__(self, img_path, df, transform=None):
        self.img_path = img_path
        self.df = df.iloc[:2500]  # Limit to first 2500 rows
        self.oct_ti = self.df['image_at_ti'].values
        self.oct_tj = self.df['image_at_ti+1'].values
        self.labels = self.df['label'].values
        self.transform = transform

        # Preload images
        self.preloaded_images = self.preload_images()

    def preload_images(self):
        preloaded = []
        for ti, tj in tqdm(zip(self.oct_ti, self.oct_tj), total=len(self.oct_ti), desc="Loading Images"):
            img_ti_path = os.path.join(self.img_path, ti)
            img_tj_path = os.path.join(self.img_path, tj)

            oct_img_ti = Image.open(img_ti_path).convert('RGB')
            oct_img_tj = Image.open(img_tj_path).convert('RGB')

            if self.transform:
                oct_img_ti = self.transform(oct_img_ti)
                oct_img_tj = self.transform(oct_img_tj)

            preloaded.append((oct_img_ti, oct_img_tj))

        return preloaded

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        oct_img_ti, oct_img_tj = self.preloaded_images[idx]
        label = self.labels[idx]
        return oct_img_ti.float(), oct_img_tj.float(), label


class SubsetDataset(CustomDataset):
    def __init__(self, original_dataset, patient_ids):
        # Copying the properties from the original dataset
        self.img_path = original_dataset.img_path
        self.transform = original_dataset.transform

        # Creating a mask based on patient IDs
        mask = original_dataset.df['id_patient'].isin(patient_ids)
        self.df = original_dataset.df[mask]

        # Update the specific attributes based on the subset of df
        self.oct_ti = self.df['image_at_ti'].values
        self.oct_tj = self.df['image_at_ti+1'].values
        self.labels = self.df['label'].values

        # Preload the images using the indices of the filtered data
        original_indices = np.where(mask)[0]
        self.preloaded_images = [original_dataset.preloaded_images[i] for i in original_indices]

    def __len__(self):
        return len(self.labels)


def specificity(true_labels, predicted_labels):
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
    FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
    TP = np.diag(conf_matrix)
    TN = conf_matrix.sum() - (FP + FN + TP)
    specificity_scores = TN / (TN + FP + 1e-6)
    return specificity_scores.mean()


def calculate_metrics(true_labels, predicted_labels):
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='micro')
    mcc = matthews_corrcoef(true_labels, predicted_labels)
    kappa = cohen_kappa_score(true_labels, predicted_labels)
    spec = specificity(true_labels, predicted_labels)
    accuracy = np.diag(conf_matrix).sum() / conf_matrix.sum()
    return conf_matrix, f1, mcc, spec, accuracy


def train_one_epoch(model, train_loader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    true_labels = []
    predicted_labels = []

    for img1, img2, labels in tqdm(train_loader):
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast():
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        probabilities = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(probabilities, dim=1)
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

    train_loss = running_loss / len(train_loader)
    train_conf_matrix, train_f1, train_mcc, train_specificity, train_accuracy = calculate_metrics(true_labels,
                                                                                                  predicted_labels)
    return train_loss, train_conf_matrix, train_f1, train_mcc, train_specificity, train_accuracy


@torch.no_grad()
def validate_one_epoch(model, valid_loader, criterion, device):
    model.eval()
    valid_loss = 0.0
    true_labels = []
    predicted_labels = []

    for img1, img2, labels in valid_loader:
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        outputs = model(img1, img2)
        loss = criterion(outputs, labels)
        valid_loss += loss.item()

        probabilities = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(probabilities, dim=1)
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

    valid_loss /= len(valid_loader)
    valid_conf_matrix, valid_f1, valid_mcc, valid_specificity, valid_accuracy = calculate_metrics(true_labels,
                                                                                                  predicted_labels)
    return valid_loss, valid_conf_matrix, valid_f1, valid_mcc, valid_specificity, valid_accuracy


def create_fold_directories(num_folds=10):
    main_dir = 'folds'
    os.makedirs(main_dir, exist_ok=True)
    for i in range(num_folds):
        os.makedirs(os.path.join(main_dir, f'fold_{i:02d}'), exist_ok=True)


# Main training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
k_folds = 20
create_fold_directories(k_folds)

# Load and preprocess the full dataset
full_df = pd.read_csv(hp['csv_train'])
full_dataset = CustomDataset(hp['img_prefix'], full_df, transform=hp['transform_valid'])

# Initial split: 90% train+val, 10% test
train_val_ids, test_ids = stratify_patients(full_df, seed=42, train_ratio=0.9, n_quartiles=2)

# Create the test dataset
test_dataset = SubsetDataset(full_dataset, patient_ids=test_ids)
test_loader = DataLoader(test_dataset, batch_size=hp['batch_size'], shuffle=False, num_workers=hp['num_workers'],
                         pin_memory=True)

# Create a dataset for train+val
train_val_dataset = SubsetDataset(full_dataset, patient_ids=train_val_ids)

models = []

for fold in range(k_folds):
        gc.collect()
        torch.cuda.empty_cache()

        run = wandb.init(
            project="MICCAI Challenge Task 1 - k-Fold Cross Validation 4",
            name=f"CustomDinoV1 t=21 - Fold {fold + 1}/{k_folds}",
            config=hp,
            group="CustomDinoV1 t=21",
            job_type=f"fold-{fold + 1}",
            reinit=True
        )

        model = hp['model'](num_classes=4).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=hp['learning_rate'], weight_decay=hp['weight_decay'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        scaler = GradScaler()

        # Split train_val into train and val for this fold
        train_ids, valid_ids = stratify_patients(train_val_dataset.df, seed=fold, train_ratio=0.9, n_quartiles=2)

        train_dataset = SubsetDataset(train_val_dataset, patient_ids=train_ids)
        valid_dataset = SubsetDataset(train_val_dataset, patient_ids=valid_ids)

        train_loader = DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=True,
                                  num_workers=hp['num_workers'],
                                  pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=hp['batch_size'], shuffle=False,
                                  num_workers=hp['num_workers'],
                                  pin_memory=True)

        best_valid_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(hp['num_epochs']):
            train_loss, train_conf_matrix, train_f1, train_mcc, train_specificity, train_accuracy = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler)
            valid_loss, valid_conf_matrix, valid_f1, valid_mcc, valid_specificity, valid_accuracy = validate_one_epoch(
                model, valid_loader, criterion, device)

            scheduler.step(valid_loss)

            wandb.log({
                'train/loss': train_loss, 'train/f1_score': train_f1, 'train/matthew_cc': train_mcc,
                'train/specificity': train_specificity, 'train/accuracy': train_accuracy,
                'valid/loss': valid_loss, 'valid/f1_score': valid_f1, 'valid/matthew_cc': valid_mcc,
                'valid/specificity': valid_specificity, 'valid/accuracy': valid_accuracy,
            })

            torch.save(model.state_dict(), os.path.join('folds', f'fold_{fold:02d}', f'epoch{epoch:02d}_model.pth'))

            if valid_loss < best_valid_loss - hp['min_delta']:
                best_valid_loss = valid_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), os.path.join('folds', f'fold_{fold:02d}', 'best_model.pth'))
                print(f"New best model saved with validation loss: {best_valid_loss:.4f}")
            else:
                epochs_without_improvement += 1

            print(f"Fold {fold + 1}/{k_folds}, Epoch {epoch + 1}/{hp['num_epochs']}, Train Loss: {train_loss:.4f}, "
                  f"Train Accuracy: {train_accuracy:.4f}, Validation Loss: {valid_loss:.4f}, "
                  f"Validation Accuracy: {valid_accuracy:.4f}")
            print(f"Validation Confusion Matrix:\n{valid_conf_matrix}")

            if epochs_without_improvement >= hp['patience']:
                print(f"Early stopping triggered. No improvement for {hp['patience']} epochs.")
                break

        # Save the best model for this fold
        best_model = hp['model'](num_classes=4).to(device)
        best_model.load_state_dict(torch.load(os.path.join('folds', f'fold_{fold:02d}', 'best_model.pth')))
        models.append(best_model)

gc.collect()
torch.cuda.empty_cache()

print("Training completed for all folds.")