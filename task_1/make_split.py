import pandas as pd
import numpy as np
from typing import Tuple, List

def stratify_patients(data: pd.DataFrame, seed=42, train_ratio: float = 0.8, n_quartiles: int = 2,
                      print_stats: bool = False) -> Tuple[List[int], List[int]]:
    """
    Stratify the patients by age group, sex, and eye side of the scans.

    Args:
    data (pd.DataFrame): DataFrame containing patient data.
    train_ratio (float): Ratio of patients to use for training (default: 0.8).
    n_quartiles (int): Number of quartiles for age and visit groups (default: 2).

    Returns:
    Tuple[List[int], List[int]]: Lists of patient IDs used for training and validation.
    """
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

def split_dataset(csv_file: str, seed=42, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15, n_quartiles: int = 2, print_stats: bool = False):
    """
    Split the dataset into training, validation, and test sets.

    Args:
    csv_file (str): Path to the CSV file containing patient data.
    seed (int): Random seed for reproducibility.
    train_ratio (float): Ratio of patients to use for training (default: 0.7).
    val_ratio (float): Ratio of patients to use for validation (default: 0.15).
    test_ratio (float): Ratio of patients to use for testing (default: 0.15).
    n_quartiles (int): Number of quartiles for age and visit groups (default: 2).
    print_stats (bool): Whether to print statistics (default: False).

    Returns:
    Tuple[List[int], List[int], List[int]]: Lists of patient IDs used for training, validation, and testing.
    """
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"

    # Load data from CSV
    data = pd.read_csv(csv_file)

    # Get train and remaining split
    train_ids, remaining_ids = stratify_patients(data, seed, train_ratio, n_quartiles, print_stats)

    # Calculate new ratios for validation and test based on the remaining data
    val_test_ratio = val_ratio / (val_ratio + test_ratio)

    # Filter the remaining data for validation and test split
    remaining_data = data[data['id_patient'].isin(remaining_ids)]

    # Stratify remaining data into validation and test sets
    val_ids, test_ids = stratify_patients(remaining_data, seed, val_test_ratio, n_quartiles, print_stats)

    return train_ids, val_ids, test_ids

if __name__ == "__main__":
    csv_file = r"D:\Git-Repositories\MARIO-Challenge\data_1\df_task1_train_challenge.csv"
    train_ids, val_ids, test_ids = split_dataset(csv_file, seed=42, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, n_quartiles=2, print_stats=True)
    print("Train IDs:", train_ids)
    print("Validation IDs:", val_ids)
    print("Test IDs:", test_ids)
