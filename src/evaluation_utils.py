import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mean_absolute_scaled_error(y_true, y_pred, y_train, m=1):
    '''
    Calculate Mean Absolute Scaled Error (MASE).
    y_true: actual values (test data)
    y_pred: predicted values (test data)
    y_train: training data
    m: seasonal period. Default is 1 (non-seasonal).
    '''
    # Compute MAE of naive prediction
    if m > 1 and len(y_train) > m:
        # MAE of seasonal naive prediction
        naive_mae = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    else:
        # MAE using naive prediction with lag-1
        naive_mae = np.mean(np.abs(np.diff(y_train)))

    if naive_mae == 0:
        return np.inf # Return infinity if denominator is zero
    
    # Compute MAE of the forecast model
    model_mae = mean_absolute_error(y_true, y_pred)
    
    return model_mae / naive_mae

# Used in each notebook
def evaluate_forecast(y_true, y_pred, y_train, m=1):
    '''
    Compute MAE, RMSE, and MASE and return them as a dictionary.
    '''
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MASE': mean_absolute_scaled_error(y_true, y_pred, y_train, m=m)
    }
    return metrics

# For consolidated evaluation results

def evaluate_forecast_result(pred_df: pd.DataFrame, y_train: pd.Series, seasonal_period: int) -> pd.Series:
    """
    Compute multiple evaluation metrics using forecast results (pred_df)
    and training data (y_train).
    
    Args:
        pred_df: DataFrame with datetime index and ['true', 'pred'] columns (test data)
        y_train: actual training values (pd.Series). Used for the denominator of classical MASE.
        seasonal_period: seasonal period (solar=48, demand=336)
        
    Returns:
        pd.Series: containing MAE, RMSE, MyIndex(RelMAE), MASE(Original)
    """
    # Copy to avoid modifying the original df
    df = pred_df.copy()
    
    # --- 1. Common preprocessing (remove missing values) ---
    # Rows with missing true or predicted values cannot be evaluated
    df_eval = df.dropna(subset=['true', 'pred'])
    
    if len(df_eval) == 0:
        return pd.Series(dtype=float)

    # --- 2. Basic metrics (MAE, RMSE) ---
    mae_model = mean_absolute_error(df_eval['true'], df_eval['pred'])
    rmse_model = np.sqrt(mean_squared_error(df_eval['true'], df_eval['pred']))

    # --- 3. MyIndex (Relative MAE) : Based on Test period ---
    # Meaning: "During this test period, how much better is the model than a simple seasonal naive forecast?"
    
    # Create seasonal naive forecast for test period
    naive_test = df['true'].shift(seasonal_period)
    
    # Identify comparable indices (where both values exist)
    valid_idx = df_eval.index.intersection(naive_test.dropna().index)
    
    if len(valid_idx) > 0:
        # Must compare at identical aligned indices; otherwise unfair
        mae_naive_test = mean_absolute_error(df.loc[valid_idx, 'true'], naive_test.loc[valid_idx])
        mae_model_aligned = mean_absolute_error(df.loc[valid_idx, 'true'], df.loc[valid_idx, 'pred'])
        
        # If denominator is nearly zero, return infinity
        my_index = mae_model_aligned / mae_naive_test if mae_naive_test > 1e-9 else np.inf
    else:
        my_index = np.nan

    # --- 4. Classical MASE : Based on Train period ---
    # Meaning: "Compared to the average variation in the training period,
    #           is the forecast error small?"
    
    if len(y_train) > seasonal_period:
        # Seasonal differencing (e.g., today's value - value from previous cycle)
        naive_train_diff = y_train.diff(seasonal_period).dropna()
        naive_train_mae = naive_train_diff.abs().mean()
        
        mase = mae_model / naive_train_mae if naive_train_mae > 1e-9 else np.inf
    else:
        mase = np.nan

    return pd.Series({
        'MAE': mae_model,
        'RMSE': rmse_model,
        'MAE(adjusted)': mae_model_aligned,
        'MAE_Naive(Test)': mae_naive_test,
        'My_Eval_Index': my_index, # original metric
        'MAE_Naive(Train)': naive_train_mae,
        'MASE (Train)': mase          # classical definition (using y_train)

    })