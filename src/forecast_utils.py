import pandas as pd
import numpy as np
from typing import Callable, Tuple, Any, Dict, Optional, List
from tqdm import tqdm
from tqdm.notebook import tqdm

# sliding window 関数　y列のみ用
def sliding_window_forecast(
    data: pd.DataFrame,
    y_col: str, 
    train_length: int, 
    test_length: int, 
    slide_count: int, 
    stride: int, 
    model_func: Callable, 
    model_params: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, List[Any], List[Any]]:
    """
    Perform sliding-window time series forecasting for a univariate target.

    Parameters
    ----------
    data : pd.DataFrame
        Input time series data.
    y_col : str
        Column name of the target variable.
    train_length : int
        Number of samples used for training in each window.
    test_length : int
        Number of samples to predict in each window.
    slide_count : int
        Number of windows to iterate.
    stride : int
        Step size for sliding the window. If None, defaults to `test_length`.
    model_func : Callable
        Function that trains a model and returns predictions.
        Expected signatures:
            - model_func(train_series, test_steps, **params)
        Return format:
            preds or (preds, model) or (preds, model, in_sample).
    model_params : dict, optional
        Additional parameters passed to model_func.

    Returns
    -------
    final_df : pd.DataFrame
        Concatenated predictions containing columns ["true", "pred"].
    models : list
        List of model objects returned from each window.
    in_samples : list
        List of in-sample information returned from each window.

    Notes
    -----
    - If prediction length does not match test_length, `np.resize` is used.
    - Stops early if data length is exceeded.
    """
    if model_params is None:
        model_params = {}
    if y_col not in data.columns:
        raise ValueError(f"{y_col} not found in DataFrame")

    series = data[y_col]
    all_preds, models, in_samples = [], [], []

    def unpack(result):
        if isinstance(result, tuple):
            preds = result[0]
            model = result[1] if len(result) > 1 else None
            insample = result[2] if len(result) > 2 else None
        else:
            preds, model, insample = result, None, None
        return np.asarray(preds), model, insample

    if stride is None:
        stride = test_length    # デフォルトは非重複ループ
    for i in tqdm(range(slide_count), desc="Sliding Window"):
        train_start = i * stride
        train_end = train_start + train_length
        test_end = train_end + test_length
        if test_end > len(series):
            print(f"Window {i+1}/{slide_count}: Data limit reached, stopping.")
            break

        train_data = series.iloc[train_start:train_end]
        test_data = series.iloc[train_end:test_end]

        preds, model, insample = unpack(
            model_func(train_data, test_steps=test_length, **model_params)
        )
        # shape mismatch対応
        if len(preds) != len(test_data):
            preds = np.resize(preds, len(test_data))

        models.append(model)
        in_samples.append(insample)

        df_i = pd.DataFrame({"true": test_data.values, "pred": preds}, index=test_data.index)
        all_preds.append(df_i)

    if not all_preds:
        return pd.DataFrame(columns=["true", "pred"]), models, in_samples

    final_df = pd.concat(all_preds)
    return final_df, models, in_samples


# 説明変数列がある場合のsliding_window関数
def sliding_window_forecast_with_features(
    data: pd.DataFrame,
    y_col: str, 
    x_cols: List[str], 
    train_length: int, 
    test_length: int, 
    slide_count: int, 
    stride: int, 
    model_func: Callable, 
    model_params: Optional[Dict[str, Any]] = None 
) -> Tuple[pd.DataFrame, List[Any], List[Any]]:
    """
    Perform sliding-window forecasting with exogenous features.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset containing target and feature columns.
    y_col : str
        Column name of the target variable.
    x_cols : list of str
        Column names of feature variables.
    train_length : int
        Number of samples used for training in each window.
    test_length : int
        Number of samples to predict in each window.
    slide_count : int
        Number of windows to iterate.
    stride : int
        Step size for sliding the window.
    model_func : Callable
        Function called as:
            model_func(X_train, y_train, X_test, **params)
        Expected to return:
            preds or (preds, model) or (preds, model, in_sample).
    model_params : dict, optional
        Additional parameters passed to the model function.

    Returns
    -------
    final_df : pd.DataFrame
        Concatenated predictions for all windows.
    models : list
        Models returned from each window.
    in_samples : list
        In-sample results from each window.

    Notes
    -----
    - Ensures all feature columns exist.
    - Stops early when data limit is reached.
    - Prediction length mismatch is resolved with `np.resize`.
    """
    # 時系列データをスライディングウィンドウで予測（特徴量対応版）。
    if model_params is None:
        model_params = {}              
    if y_col not in data.columns:
        raise ValueError(f"Target column {y_col} not found in DataFrame")
    for col in x_cols:
        if col not in data.columns:
            raise ValueError(f"Feature column {col} not found in DataFrame")

    all_preds, models, in_samples = [], [], []

    def unpack(result):
        if isinstance(result, tuple):
            preds, model, insample = result[0], (result[1] if len(result) > 1 else None), (result[2] if len(result) > 2 else None)
        else:
            preds, model, insample = result, None, None
        return np.asarray(preds), model, insample

    for i in tqdm(range(slide_count), desc="Sliding Window"):
        train_start = i * stride
        train_end = train_start + train_length
        test_end = train_end + test_length
        if test_end > len(data):
            print(f"Window {i+1}/{slide_count}: Data limit reached, stopping.")
            break

        train_df = data.iloc[train_start:train_end]
        test_df = data.iloc[train_end:test_end]

        X_train, y_train = train_df[x_cols], train_df[y_col]
        X_test, y_test = test_df[x_cols], test_df[y_col]

        preds, model, insample = unpack(model_func(
            X_train, y_train, X_test, 
            **model_params
            ))

        if len(preds) != len(y_test):
            preds = np.resize(preds, len(y_test))

        models.append(model)
        in_samples.append(insample)
        df_i = pd.DataFrame({"true": y_test.values, "pred": preds}, index=y_test.index)
        all_preds.append(df_i)

    if not all_preds:
        return pd.DataFrame(columns=["true", "pred"]), [], []

    return pd.concat(all_preds), models, in_samples

