import gc
import os

import torch
from PIL import Image
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np

# Assume Models and Transforms are imported from separate files
from Models import CustomDinoV1
from Transforms import transformV21

# Constants
DATA_DIR = os.path.join('..', 'data_1')
VAL_CSV = os.path.join(DATA_DIR, 'df_task1_val_challenge.csv')
IMG_PREFIX = os.path.join(DATA_DIR, 'data_task1', 'val')
FOLDS_DIR = 'folds'

class PreloadedInferenceDataset(Dataset):
    def __init__(self, img_path, df, transform=None):
        self.df = df
        self.oct_ti = df['image_at_ti'].values
        self.oct_tj = df['image_at_ti+1'].values
        self.transform = transform
        self.images = self.preload_images(img_path)

    def preload_images(self, img_path):
        images = {}
        unique_images = set(self.oct_ti) | set(self.oct_tj)
        for img_name in tqdm(unique_images, desc="Preloading images"):
            img_path_full = os.path.join(img_path, img_name)
            img = Image.open(img_path_full).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images[img_name] = img
        return images

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        oct_img_ti = self.images[self.oct_ti[idx]]
        oct_img_tj = self.images[self.oct_tj[idx]]
        return oct_img_ti.float(), oct_img_tj.float()

def predict_k_folds(csv_file, img_dir, folds_dir, model_class, transform, batch_size=24, num_workers=0, device='cuda'):
    # Load the original CSV
    df = pd.read_csv(csv_file)

    # Create dataset and dataloader
    dataset = PreloadedInferenceDataset(img_dir, df, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Get the number of folds
    num_folds = len([f for f in os.listdir(folds_dir) if f.startswith('fold_')])

    # Make predictions
    all_predictions = np.zeros(shape=(len(df), num_folds, 4))

    with torch.no_grad(), autocast():
        for m in range(num_folds):
            # Load model for current fold
            model_path = os.path.join(folds_dir, f'fold_{m:02d}', 'best_model.pth')
            model = model_class(num_classes=4).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            fold_predictions = []
            for img1, img2 in tqdm(dataloader, desc=f"Fold {m + 1}/{num_folds}"):
                img1, img2 = img1.to(device), img2.to(device)
                outputs = model(img1, img2)
                fold_predictions.append(outputs.cpu().numpy())

            # Concatenate all predictions for this fold
            fold_predictions = np.concatenate(fold_predictions, axis=0)
            all_predictions[:, m, :] = fold_predictions

            # Clear CUDA cache after processing each fold
            if device == 'cuda':
                gc.collect()
                torch.cuda.empty_cache()

    # Add predictions to the dataframe
    df['k_fold_predictions'] = [all_predictions[i].tolist() for i in range(len(df))]

    return df

# Main execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run predictions
    results_df = predict_k_folds(
        csv_file=VAL_CSV,
        img_dir=IMG_PREFIX,
        folds_dir=FOLDS_DIR,
        model_class=CustomDinoV1,
        transform=transformV21,
        batch_size=20,
        num_workers=0,
        device=device
    )

    # Save results
    results_df.to_csv('validation_predictions.csv', index=False)
    print("Predictions saved to 'validation_predictions.csv'")


import pandas as pd
import numpy as np

# Load the predictions CSV
predictions_csv = 'validation_predictions.csv'
df = pd.read_csv(predictions_csv)

# Apply softmax function row-wise
def softmax(x):
    # Subtract max per row for numerical stability
    x = x - np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=1, keepdims=True)

# Function to process predictions
def process_predictions(row):
    predictions = softmax(np.array(eval(row['k_fold_predictions'])))  # Convert string to numpy array
    sum_predictions = predictions.sum(axis=0)  # Sum across all folds
    return np.argmax(sum_predictions)  # Return the class with the highest sum

# Apply the processing function to each row
df['final_prediction'] = df.apply(process_predictions, axis=1)

# Prepare the submission data
submission_data = np.column_stack((df['case'], df['final_prediction']))

# Save the submission file
np.savetxt('submission_predictions.csv',
           submission_data,
           delimiter=',',
           fmt='%s',
           header='case,prediction',
           comments='')

# Load the predictions CSV
predictions_csv = 'validation_predictions.csv'
df = pd.read_csv(predictions_csv)

# Apply softmax function row-wise
def softmax(x):
    # Subtract max per row for numerical stability
    x = x - np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=1, keepdims=True)

# Function to process predictions
def process_predictions(row):
    predictions = softmax(np.array(eval(row['k_fold_predictions'])))  # Convert string to numpy array
    sum_predictions = predictions.sum(axis=0)  # Sum across all folds
    return np.argmax(sum_predictions)  # Return the class with the highest sum

# Apply the processing function to each row
df['final_prediction'] = df.apply(process_predictions, axis=1)

# Prepare the submission data
submission_data = np.column_stack((df['case'], df['final_prediction']))

# Save the submission file
np.savetxt('submission_predictions.csv',
           submission_data,
           delimiter=',',
           fmt='%s',
           header='case,prediction',
           comments='')

print("Submission file 'submission_predictions.csv' has been created.")