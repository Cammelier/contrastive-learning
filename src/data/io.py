from typing import Tuple
from pathlib import Path
import pickle
import logging

import pandas as pd


def load_df(file_path: str, **kwargs) -> pd.DataFrame:
    ext = Path(file_path).suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(file_path, **kwargs)
    elif ext == ".csv":
        return pd.read_csv(file_path, **kwargs)
    elif ext == ".pkl" or ext == ".pickle":
        with open(file_path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def save_df(
    df: pd.DataFrame,
    file_path: str,
    index: bool = False,
    **kwargs,
) -> None:
    ext = Path(file_path).suffix.lower()
    if ext == ".parquet":
        df.to_parquet(file_path, index=index, **kwargs)
    elif ext == ".csv":
        df.to_csv(file_path, index=index, **kwargs)
    elif ext == ".pkl" or ext == ".pickle":
        with open(file_path, "wb") as f:
            pickle.dump(df, f)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


logger = logging.getLogger(__name__)


def load_data_splits(
    base_path: Path, file_base: str, ext: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test dataframes.

    Args:
        base_path: Path to the directory containing data files
        file_base: Base name of the data files
        ext: File extension

    Returns:
        Tuple of (train_df, val_df, test_df)

    Raises:
        FileNotFoundError: If data files are not found
        Exception: For other loading errors
    """
    try:
        train_df = load_df(base_path / f"{file_base}_train.{ext}")
        val_df = load_df(base_path / f"{file_base}_val.{ext}")
        test_df = load_df(base_path / f"{file_base}_test.{ext}")
        return train_df, val_df, test_df
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data files: {e}")
        raise
