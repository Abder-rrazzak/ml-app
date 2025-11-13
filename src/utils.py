"""Utility functions for the ML project."""

import os
import pandas as pd


def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)


def save_data(data, filepath):
    """Save DataFrame to CSV file."""
    ensure_dir(os.path.dirname(filepath))
    data.to_csv(filepath, index=False)


def load_data(filepath):
    """Load data from CSV file."""
    return pd.read_csv(filepath)
