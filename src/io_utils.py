"""
Utility functions for loading and saving membrane simulation data (CSV/JSON).

Provides functions to load and save data in CSV and JSON formats using pandas and json libraries.
"""
import pandas as pd
import json

def load_csv(filepath):
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(filepath)

def save_csv(df, filepath):
    """Save a pandas DataFrame to a CSV file."""
    df.to_csv(filepath, index=False)

def load_json(filepath):
    """Load a JSON file into a Python object."""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json(data, filepath):
    """Save a Python object as a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
