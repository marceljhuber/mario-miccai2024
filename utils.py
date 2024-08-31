import pandas as pd
import torch
import numpy as np
import random
from typing import List, Tuple

def set_random_seeds(seed_value):
    # Set seed for Python's built-in random module
    random.seed(seed_value)
    
    # Set seed for numpy
    np.random.seed(seed_value)
    
    # Set seed for torch
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    
    # To ensure that the CUDA operations are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stratify_patients(csv_file: str, seed=42, train_ratio: float = 0.8, n_quartiles: int = 2,
                      print_stats: bool = False) -> Tuple[List[int], List[int]]:
    """
    Stratify the patients by age group, sex, and eye side of the scans.

    Args:
    csv_file (str): Path to the CSV file containing patient data.
    train_ratio (float): Ratio of patients to use for training (default: 0.8).
    n_quartiles (int): Number of quartiles for age and visit groups (default: 2).

    Returns:
    Tuple[List[int], List[int]]: Lists of patient IDs used for training and validation.
    """
    # Load data from CSV
    data = pd.read_csv(csv_file)

    # Set the random split
    np.random.seed(seed)

    stratified = data.copy()
    stratified["is_male"] = stratified["sex"] == "M"  # assuming static gender

    # Use these for stratification
    aggregations = {
        "age_at_ti": "mean",
        "num_current_visit_at_i": "max",
        "is_male": "first",
    }

    stratified = stratified.groupby("id_patient").agg(aggregations).reset_index()

    # Split into age groups with ~equal number of patients
    stratified["age_group"] = pd.qcut(
        stratified["age_at_ti"], q=n_quartiles, duplicates="drop"
    )
    stratified["max_visits_group"] = pd.qcut(
        stratified["num_current_visit_at_i"], q=n_quartiles, duplicates="drop"
    )

    if print_stats:
        print("Stratification age groups: ", stratified["age_group"].value_counts())
        print("Stratification visit groups: ", stratified["max_visits_group"].value_counts())
        print("Stratification male ratio:", stratified["is_male"].mean())

    grouped = stratified.groupby(
        ["is_male", "age_group", "max_visits_group"], observed=True
    )

    train_ids = []
    valid_ids = []

    for _, group in grouped:
        unique_patients = group["id_patient"].unique()
        np.random.shuffle(unique_patients)
        # Biases the split towards the valid set
        split_point = int(len(unique_patients) * train_ratio)
        train_ids.extend(unique_patients[:split_point])
        valid_ids.extend(unique_patients[split_point:])

    assert set(train_ids).intersection(set(valid_ids)) == set(), "Overlap in splits!"

    return train_ids, valid_ids
