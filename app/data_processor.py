import pandas as pd
import numpy as np
import time
from app.data_handler import load_csv
from app.optimizer import run_optimizer

# =============================================================================
# DATA PROCESSOR FOR TRADING STRATEGY OPTIMIZATION
# =============================================================================
#
# Processes datasets and runs the optimizer.
# Calls `run_optimizer()` from `app.optimizer` to optimize the trading strategy.
# =============================================================================

def create_hourly_predictions(df, horizon):
    """
    Auto-compute hourly predictions from the base dataset.
    """
    blocks = []
    for i in range(len(df) - horizon):
        block = df.iloc[i+1 : i+1+horizon].values.flatten()
        blocks.append(block)
    return pd.DataFrame(blocks, index=df.index[:-horizon])

def create_daily_predictions(df, horizon):
    """
    Auto-compute daily predictions from the base dataset.

    Each row i in 'df' yields a block of predicted values at offsets t+24, t+48, ... t+24*horizon.
    If there are not enough rows (or any row indexing is invalid), we return an empty DataFrame.
    """
    nrows = len(df)
    required_rows = horizon * 24

    # If the dataset is too short for daily predictions, return empty.
    if nrows < required_rows:
        print("Warning: Not enough rows to create daily predictions. Returning an empty DataFrame.")
        return pd.DataFrame()

    blocks = []
    # For each row i, gather the next horizon days (24*horizon ticks).
    for i in range(nrows - required_rows):
        block_values = []
        # Attempt to fetch each day offset at i + d*24
        for d in range(1, horizon + 1):
            idx = i + d * 24
            # If idx is out of range for some reason (edge cases), skip safely
            if idx >= nrows:
                break
            row_vals = df.iloc[idx].values  # shape: (features,)
            if row_vals.ndim == 0:
                # means we have an empty row or zero-dimensional array
                continue
            block_values.extend(row_vals.flatten())
        # If we got some values, append them to blocks
        if block_values:
            blocks.append(block_values)
        else:
            # If we ended up with no valid data for that block, skip
            continue

    if not blocks:
        # No valid blocks were created
        print("Warning: daily predictions are empty after alignment. Returning an empty DataFrame.")
        return pd.DataFrame()

    # The resulting blocks length = nrows - required_rows
    # Align the index accordingly
    daily_idx = df.index[:(nrows - required_rows)]
    return pd.DataFrame(blocks, index=daily_idx)



def process_data(config):
    """
    Loads and processes datasets.
    
    - Loads the hourly predictions, daily predictions (if provided), and the base dataset.
    - If 'max_steps' is provided in config, each dataset is truncated to the first max_steps rows.
    - If a predictions file is not provided, predictions are auto-generated using config["time_horizon"].
    - If a date_column is provided, it is converted to a datetime index (named "DATE_TIME").
    - Finally, the datasets are aligned by their common date range and numeric conversion is applied.
    
    Args:
        config (dict): Configuration with keys including:
            - "hourly_predictions_file"
            - "daily_predictions_file"
            - "base_dataset_file"
            - "headers"
            - "date_column"
            - "time_horizon"
            - "max_steps"
    
    Returns:
        dict: Dictionary with keys "hourly", "daily", "base" holding the processed DataFrames.
    """
    headers = config.get("headers", True)
    print("Loading datasets...")

    hourly_df = load_csv(config["hourly_predictions_file"], headers=headers) if config.get("hourly_predictions_file") else None
    daily_df = load_csv(config["daily_predictions_file"], headers=headers) if config.get("daily_predictions_file") else None
    base_df = load_csv(config["base_dataset_file"], headers=headers)

    print(f"Base dataset loaded: {base_df.shape}")

    # Apply max_steps truncation if specified
    max_steps = config.get("max_steps")
    if max_steps is not None:
        if hourly_df is not None:
            hourly_df = hourly_df.iloc[:max_steps]
        if daily_df is not None:
            daily_df = daily_df.iloc[:max_steps]
        base_df = base_df.iloc[:max_steps]
        print(f"Datasets truncated to the first {max_steps} rows.")

    if hourly_df is None:
        if "time_horizon" not in config or not config["time_horizon"]:
            raise ValueError("time_horizon must be provided when auto-generating predictions.")
        print("Auto-generating hourly predictions...")
        hourly_df = create_hourly_predictions(base_df, config["time_horizon"])

    if daily_df is None:
        if "time_horizon" not in config or not config["time_horizon"]:
            raise ValueError("time_horizon must be provided when auto-generating predictions.")
        print("Auto-generating daily predictions...")
        daily_df = create_daily_predictions(base_df, config["time_horizon"])

    print("\nDatasets loaded:")
    print(f"  Hourly predictions: {hourly_df.shape}")
    print(f"  Daily predictions:  {daily_df.shape}")
    print(f"  Base dataset:       {base_df.shape}")

    # Convert date_column to datetime index if provided
    date_column = config.get("date_column")
    if date_column:
        for label, df in zip(["Hourly", "Daily", "Base"], [hourly_df, daily_df, base_df]):
            if date_column in df.columns:
                df[date_column] = pd.to_datetime(df[date_column])
                df.set_index(date_column, inplace=True)
                df.index.name = "DATE_TIME"
            else:
                print(f"Warning: '{date_column}' not found in {label} dataset.")

    # Align datasets by common date range
    try:
        common_start = max(hourly_df.index.min(), daily_df.index.min(), base_df.index.min())
        common_end = min(hourly_df.index.max(), daily_df.index.max(), base_df.index.max())
        hourly_df = hourly_df.loc[common_start:common_end]
        daily_df = daily_df.loc[common_start:common_end]
        base_df = base_df.loc[common_start:common_end]
        print(f"Datasets aligned to common date range: {common_start} to {common_end}")
    except Exception as e:
        print("Error aligning datasets by date:", e)

    # Convert all columns to numeric and fill missing values
    for label, df in zip(["Hourly", "Daily", "Base"], [hourly_df, daily_df, base_df]):
        df[df.columns] = df[df.columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        print(f"{label} dataset: Converted columns to numeric (final shape: {df.shape}).")

    return {"hourly": hourly_df, "daily": daily_df, "base": base_df}


def run_processing_pipeline(config, plugin):
    """
    Executes the trading strategy optimization pipeline.
    Now writes out the final balance plot (renamed), the trades CSV, and a summary CSV.
    Optionally logs the config['strategy_name'] if present.
    """

    start_time = time.time()
    strat_name = config.get("strategy_name", "Heuristic Strategy")
    print(f"\n=== Starting Trading Strategy Optimization Pipeline for '{strat_name}' ===")

    datasets = process_data(config)
    hourly_preds, daily_preds, base_data = datasets["hourly"], datasets["daily"], datasets["base"]

    print("\nProcessed Dataset Shapes:")
    print(f"  Hourly predictions: {hourly_preds.shape}")
    print(f"  Daily predictions:  {daily_preds.shape}")
    print(f"  Base rates:         {base_data.shape}")

    if hasattr(plugin, "get_optimizable_params") and hasattr(plugin, "evaluate_candidate"):
        print(f"\nPlugin supports optimization. Running optimizer for '{strat_name}'...")
        trading_info = run_optimizer(plugin, base_data, hourly_preds, daily_preds, config)
    else:
        print("\nPlugin does not support optimization. Exiting.")
        trading_info = {}

    print("\n=== Optimization Results ===")
    for key, value in trading_info.items():
        print(f"{key}: {value}")

    # --------------------------------------------------
    # (1) Rename the final balance plot if it was created
    # The plugin's strategy "stop()" typically saves 'balance_plot.png'.
    # We rename it if 'balance_plot_file' is given.
    # --------------------------------------------------
    if config.get("balance_plot_file"):
        import os
        old_plot = "balance_plot.png"
        new_plot = config["balance_plot_file"]
        if os.path.exists(old_plot):
            try:
                os.rename(old_plot, new_plot)
                print(f"Renamed {old_plot} -> {new_plot}")
            except Exception as e:
                print(f"Failed to rename {old_plot} to {new_plot}: {e}")
        else:
            print(f"Warning: {old_plot} not found; no balance plot to rename.")

    # --------------------------------------------------
    # (2) Save trades to CSV if 'trades_csv_file' is specified
    # We assume plugin's strategy stores trades in plugin.trades or plugin.model.trades
    # but usually it's in plugin trades if the plugin duplicates them.
    # If not, you can store them from the strategy instance in evaluate_candidate.
    # --------------------------------------------------
    trades_csv = config.get("trades_csv_file")
    if trades_csv:
        try:
            import pandas as pd
            # If your plugin or strategy saves trades in plugin.trades, do that:
            if hasattr(plugin, "trades") and plugin.trades:
                pd.DataFrame(plugin.trades).to_csv(trades_csv, index=False)
                print(f"Trades saved to {trades_csv}.")
            else:
                print("Warning: plugin.trades not found or empty.")
        except Exception as e:
            print(f"Failed to save trades to {trades_csv}: {e}")

    # --------------------------------------------------
    # (3) Save summary CSV if 'summary_csv_file' is specified
    # We'll store final trading_info plus optionally any plugin variables
    # --------------------------------------------------
    summary_csv = config.get("summary_csv_file")
    if summary_csv:
        try:
            import pandas as pd
            df = pd.DataFrame([trading_info])
            df.to_csv(summary_csv, index=False)
            print(f"Summary saved to {summary_csv}.")
        except Exception as e:
            print(f"Failed to save summary CSV to {summary_csv}: {e}")

    end_time = time.time()
    print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")
    return trading_info, getattr(plugin, "trades", None)


if __name__ == "__main__":
    pass
