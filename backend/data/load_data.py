import pandas as pd
import seaborn as sns 


def load_data():
    """Load the dataset and return a pandas DataFrame."""
    df = sns.load_dataset("tips")
    return df
