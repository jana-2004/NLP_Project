"""
Data loading and preprocessing for mental health sentiment analysis.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Tuple


def load_and_process_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, LabelEncoder]:
    """
    Load the raw CSV and return processed train/test splits and a fitted label encoder.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        Tuple of (train_df, test_df, label_encoder)
    """
    df = pd.read_csv(file_path)

    # Drop NaNs
    df = df.dropna(subset=["statement", "status"])

    # Encode labels
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["status"])

    # Train-test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

    # Reset index to fix KeyError bug in Dataset class
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df, label_encoder
