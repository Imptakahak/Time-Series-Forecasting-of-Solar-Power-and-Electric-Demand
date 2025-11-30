from pathlib import Path
import pandas as pd

def load_timeseries_data(file_path: str) -> pd.DataFrame:
    """
    Generic function to load any time series CSV file.
    Assumes publicly available, high-quality data.

    Args:
        file_path (str): Full path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"The specified file does not exist: {file_path}")

    df = pd.read_csv(file_path, parse_dates=["datetime"], index_col="datetime")
    return df


def validate_no_missing(df: pd.DataFrame) -> None:
    """
    Validate that there are no missing values.

    Args:
        df (pd.DataFrame): DataFrame to be checked.

    Raises:
        ValueError: Raised when missing values are found.
    """
    na_counts = df.isna().sum()
    total_missing = na_counts.sum()

    if total_missing > 0:
        cols_with_na = na_counts[na_counts > 0]
        msg = "Columns containing missing values:\n" + cols_with_na.to_string()
        raise ValueError(msg)

    print("Missing value check completed: No missing values in any column")

