import datetime
import os
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy

class Plugin:
    """
    Plugin for Heuristic Trading Strategy.

    - Implements a heuristic trading strategy with configurable parameters.
    - The strategy uses short-term (hourly) and long-term (daily) predictions to:
        1. Calculate potential profit and risk for both long and short orders.
        2. Compare their profit/risk ratios and only enter a trade if the ratio exceeds
           a configurable threshold.
        3. Set TP (Take Profit) and SL (Stop Loss) levels based on the extremes found in the long-term predictions.
        4. Continuously monitor the market (each tick, assumed hourly) and exit the trade if:
           - For a long: the price reaches TP, or if the predicted minimum (from short+long) falls below SL.
           - For a short: the price reaches TP, or if the predicted maximum rises above SL.
    - Designed to operate one order at a time.
    - The parameters are intended to be optimized (using a genetic algorithm, for example).
    """

    # Default plugin parameters (for optimizer integration)
    plugin_params = {
        'pip_cost': 0.00001,
        'rel_volume': 0.02,         # Maximum 2% of available balance
        'min_order_volume': 10000,
        'max_order_volume': 1000000,
        'leverage': 1000,
        'profit_threshold': 5,      # Minimum profit (in pips) to consider a trade
        'min_drawdown_pips': 10,
        'tp_multiplier': 0.9,
        'sl_multiplier': 2.0,
        'lower_rr_threshold': 0.5,
        'upper_rr_threshold': 2.0,
        'time_horizon': 3           # In days; used to compute expected TP tick (3 days = 72 ticks)
    }

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.trades = []

    def set_params(self, **kwargs):
        """Update plugin parameters dynamically."""
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def get_debug_info(self):
        """Return debugging information for the plugin."""
        return {key: self.params[key] for key in self.params}

    # -------------------------------------------------------------------------
    # Optimization Interface
    # -------------------------------------------------------------------------
    def get_optimizable_params(self):
        """Return parameters that can be optimized along with their bounds."""
        return [
            ("pip_cost", 0.000001, 0.0001),
            ("rel_volume", 0.01, 0.1),
            ("min_order_volume", 1000, 100000),
            ("max_order_volume", 100000, 2000000),
            ("leverage", 1, 5000),
            ("profit_threshold", 0.5, 20),
            ("min_drawdown_pips", 1, 50),
            ("tp_multiplier", 0.5, 1.5),
            ("sl_multiplier", 1.5, 6.0),
            ("lower_rr_threshold", 0.2, 1.0),
            ("upper_rr_threshold", 1.3, 6.0),
            ("time_horizon", 2, 24)
        ]

    def evaluate_candidate(self, individual, base_data, hourly_predictions, daily_predictions, config):
        import os
        import pandas as pd
        from backtesting import Backtest

        # Unpack candidate parameters:
        # If individual has 12 values, use them all; if it has 6, use the optimized ones and take default for the rest.
        if len(individual) == 12:
            pip_cost, rel_volume, min_order_volume, max_order_volume, leverage, profit_threshold, \
            min_drawdown_pips, tp_multiplier, sl_multiplier, lower_rr, upper_rr, time_horizon = individual
        elif len(individual) == 6:
            profit_threshold, tp_multiplier, sl_multiplier, lower_rr, upper_rr, time_horizon = individual
            pip_cost = self.params['pip_cost']
            rel_volume = self.params['rel_volume']
            min_order_volume = self.params['min_order_volume']
            max_order_volume = self.params['max_order_volume']
            leverage = self.params['leverage']
            min_drawdown_pips = self.params['min_drawdown_pips']
        else:
            raise ValueError(f"Expected candidate with 6 or 12 values, got {len(individual)}")

        # Update plugin parameters
        self.params.update({
            'pip_cost': pip_cost,
            'rel_volume': rel_volume,
            'min_order_volume': min_order_volume,
            'max_order_volume': max_order_volume,
            'leverage': leverage,
            'profit_threshold': profit_threshold,
            'min_drawdown_pips': min_drawdown_pips,
            'tp_multiplier': tp_multiplier,
            'sl_multiplier': sl_multiplier,
            'lower_rr_threshold': lower_rr,
            'upper_rr_threshold': upper_rr,
            'time_horizon': int(time_horizon)
        })

        # Auto-generate predictions if none provided.
        if (config.get('hourly_predictions_file') is None) and (config.get('daily_predictions_file') is None):
            print(f"[evaluate_candidate] Auto-generating predictions using time_horizon={int(time_horizon)} for candidate {individual}.")
            from data_processor import process_data
            processed = process_data(config)
            hourly_predictions = processed["hourly"]
            daily_predictions = processed["daily"]

        # Merge predictions into one DataFrame.
        merged_df = pd.DataFrame()
        if hourly_predictions is not None and not hourly_predictions.empty:
            renamed_h = {col: f"Prediction_h_{i+1}" for i, col in enumerate(hourly_predictions.columns)}
            hr = hourly_predictions.rename(columns=renamed_h)
            merged_df = hr.copy() if merged_df.empty else merged_df.join(hr, how="outer")
        if daily_predictions is not None and not daily_predictions.empty:
            renamed_d = {col: f"Prediction_d_{i+1}" for i, col in enumerate(daily_predictions.columns)}
            dr = daily_predictions.rename(columns=renamed_d)
            merged_df = dr.copy() if merged_df.empty else merged_df.join(dr, how="outer")
        if merged_df.empty:
            print(f"[evaluate_candidate] => Merged predictions are empty for candidate {individual}. Returning profit=0.0.")
            return (0.0, {"num_trades": 0, "win_pct": 0, "max_dd": 0, "sharpe": 0})
        merged_df.index.name = "DATE_TIME"
        temp_pred_file = "temp_predictions.csv"
        merged_df.reset_index().to_csv(temp_pred_file, index=False)

        # Normalize base_data columns required by backtesting.py.
        required_cols = ['Open', 'High', 'Low', 'Close']
        for col in required_cols:
            if col not in base_data.columns and col.upper() in base_data.columns:
                base_data.rename(columns={col.upper(): col}, inplace=True)
        for col in required_cols:
            if col not in base_data.columns:
                if col != 'Close' and 'Close' in base_data.columns:
                    print(f"[evaluate_candidate] Column '{col}' missing. Creating it using 'Close'.", flush=True)
                    base_data[col] = base_data['Close']
                else:
                    raise ValueError(f"Base data must contain '{col}' column.")

        # Ensure base_data index is a DatetimeIndex.
        base_data = base_data.copy()
        if not isinstance(base_data.index, pd.DatetimeIndex):
            base_data.index = pd.to_datetime(base_data.index)

        # Prepare extra info to pass to the strategy.
        extra = {"hourly": hourly_predictions, "daily": daily_predictions, "params_config": self.params}

        # Instead of attaching extra to base_data (which backtesting.py does not support), we attach it to the Backtest instance.
        bt_sim = Backtest(base_data, self.HeuristicStrategy, cash=10000, commission=0.0, exclusive_orders=True)
        bt_sim.extra = extra  # <-- Store extra info in the Backtest instance
        perf = bt_sim.run()
        final_balance = perf["Equity"].iloc[-1]
        profit = final_balance - 10000.0
        print(f"[BACKTEST ANALYZE] Final Balance: {final_balance:.2f} | Profit: {profit:.2f}", flush=True)
        if os.path.exists(temp_pred_file):
            os.remove(temp_pred_file)
        return (profit, {"portfolio": perf})


    # -------------------------------------------------------------------------
    # Strategy Implementation using backtesting.py
    # -------------------------------------------------------------------------
    class HeuristicStrategy(Strategy):
        """
        Heuristic Trading Strategy.
        
        Entry:
        - Uses short-term (hourly) and long-term (daily) predictions to compute:
            • Profit potential for long: (max_daily - current_price) / pip_cost.
            • Risk for long: max( (current_price - min_daily) / pip_cost, min_drawdown_pips ).
            • Profit potential for short: (current_price - min_daily) / pip_cost.
            • Risk for short: max( (max_daily - current_price) / pip_cost, min_drawdown_pips ).
        - Compares profit/risk (RR) of long and short; enters the trade (long or short) with the better RR if the profit exceeds a threshold.
        - Order volume is set proportionally between min_order_volume and max_order_volume.
        
        Exit:
        - At each tick (assumed hourly), monitors market behavior.
        - For a long position: closes if price reaches TP or if the predictions (from daily data) indicate that SL would be hit.
        - For a short position: closes if price reaches TP or if predictions indicate that SL would be hit.
        """
        def init(self):
            # Retrieve extra information passed from Backtest (attached to bt_sim.extra)
            self.extra = self.bt.extra  # Contains 'hourly', 'daily', and 'params_config'
            # Initialize trade tracking variables.
            self.trade_entry_bar = None
            self.trade_low = None
            self.trade_high = None
            self.entry_profit = None
            self.entry_risk = None
            self.entry_rr = None
            self.entry_signal = None
            # Store the order entry price (will be set in notify_order below).
            self.order_entry_price = None

        def next(self):
            # Get current price and datetime.
            current_price = self.data.Close[-1]
            dt = self.data.index[-1]

            # If a position is open, evaluate exit conditions.
            if self.position:
                if self.entry_signal == 'long':
                    if (self.trade_low is None) or (current_price < self.trade_low):
                        self.trade_low = current_price
                    # For exit we use a simplified approach: if price reaches TP or falls below SL, exit.
                    if current_price >= self.current_tp:
                        self.position.close()
                        return
                    if current_price < self.current_sl:
                        self.position.close()
                        return
                elif self.entry_signal == 'short':
                    if (self.trade_high is None) or (current_price > self.trade_high):
                        self.trade_high = current_price
                    if current_price <= self.current_tp:
                        self.position.close()
                        return
                    if current_price > self.current_sl:
                        self.position.close()
                        return
                return  # Do not attempt new entries if a position is open.

            # No position open: update extremes.
            self.trade_low = current_price
            self.trade_high = current_price

            # Get the extra prediction info (we use the daily predictions for entry extremes)
            extra = self.extra  # Dictionary with keys "daily" and "hourly"
            daily_preds_df = extra["daily"]
            # Try to locate the row corresponding to the current datetime; if not found, use the last row.
            try:
                row = daily_preds_df.loc[dt]
            except KeyError:
                row = daily_preds_df.iloc[-1]
            # Extract all daily prediction values from columns that start with "Prediction_d_"
            daily_preds = [row[col] for col in row.index if col.startswith("Prediction_d_")]
            if not daily_preds:
                return
            max_pred = max(daily_preds)
            min_pred = min(daily_preds)

            # Calculate entry conditions for LONG.
            ideal_profit_pips_buy = (max_pred - current_price) / self.params.pip_cost
            ideal_drawdown_pips_buy = max((current_price - min_pred) / self.params.pip_cost, self.params.min_drawdown_pips)
            rr_buy = ideal_profit_pips_buy / ideal_drawdown_pips_buy if ideal_drawdown_pips_buy > 0 else 0
            tp_buy = current_price + self.params.tp_multiplier * ideal_profit_pips_buy * self.params.pip_cost
            sl_buy = current_price - self.params.sl_multiplier * ideal_drawdown_pips_buy * self.params.pip_cost

            # Calculate entry conditions for SHORT.
            ideal_profit_pips_sell = (current_price - min_pred) / self.params.pip_cost
            ideal_drawdown_pips_sell = max((max_pred - current_price) / self.params.pip_cost, self.params.min_drawdown_pips)
            rr_sell = ideal_profit_pips_sell / ideal_drawdown_pips_sell if ideal_drawdown_pips_sell > 0 else 0
            tp_sell = current_price - self.params.tp_multiplier * ideal_profit_pips_sell * self.params.pip_cost
            sl_sell = current_price + self.params.sl_multiplier * ideal_drawdown_pips_sell * self.params.pip_cost

            # Print entry debug information.
            print(f"[DEBUG ENTRY] {dt} | Current Price: {current_price:.5f}", flush=True)
            print(f"[DEBUG ENTRY - LONG] Profit: {ideal_profit_pips_buy:.2f} pips, Risk: {ideal_drawdown_pips_buy:.2f} pips, RR: {rr_buy:.2f}, TP: {tp_buy:.5f}, SL: {sl_buy:.5f}", flush=True)
            print(f"[DEBUG ENTRY - SHORT] Profit: {ideal_profit_pips_sell:.2f} pips, Risk: {ideal_drawdown_pips_sell:.2f} pips, RR: {rr_sell:.2f}, TP: {tp_sell:.5f}, SL: {sl_sell:.5f}", flush=True)

            # Determine which side to enter.
            if (ideal_profit_pips_buy >= self.params.profit_threshold) and (rr_buy >= rr_sell):
                signal = 'long'
                self.current_tp = tp_buy
                self.current_sl = sl_buy
            elif (ideal_profit_pips_sell >= self.params.profit_threshold) and (rr_sell > rr_buy):
                signal = 'short'
                self.current_tp = tp_sell
                self.current_sl = sl_sell
            else:
                return  # Do not enter trade if minimum profit condition is not met.

            # Store entry metrics for later debug in trade exit.
            self.entry_profit = ideal_profit_pips_buy if signal == 'long' else ideal_profit_pips_sell
            self.entry_risk = ideal_drawdown_pips_buy if signal == 'long' else ideal_drawdown_pips_sell
            self.entry_rr = rr_buy if signal == 'long' else rr_sell
            self.entry_signal = signal

            # Calculate order size (proportional allocation between min and max order volumes)
            order_size = self.compute_size(self.entry_rr)
            if order_size <= 0:
                print("[DEBUG] Order size <= 0, skipping trade", flush=True)
                return

            # Record the entry bar for later duration calculation.
            self.trade_entry_bar = len(self)
            # Execute the order.
            if signal == 'long':
                self.buy(size=order_size)
                print(f"[DEBUG ENTRY] LONG order executed. Order Size: {order_size}", flush=True)
            elif signal == 'short':
                self.sell(size=order_size)
                print(f"[DEBUG ENTRY] SHORT order executed. Order Size: {order_size}", flush=True)

        def compute_size(self, rr):
            min_vol = self.params.min_order_volume
            max_vol = self.params.max_order_volume
            if rr >= self.params.upper_rr_threshold:
                size = max_vol
            elif rr <= self.params.lower_rr_threshold:
                size = min_vol
            else:
                size = min_vol + ((rr - self.params.lower_rr_threshold) /
                                (self.params.upper_rr_threshold - self.params.lower_rr_threshold)) * (max_vol - min_vol)
            cash = self.broker.cash
            max_from_cash = cash * self.params.rel_volume * self.params.leverage
            return min(size, max_from_cash)

        def notify_order(self, order):
            if order.status == order.Completed:
                self.order_entry_price = order.executed.price

        def on_trade(self, trade):
            # Called when a trade is closed.
            if trade.isclosed:
                duration = len(self) - (self.trade_entry_bar if self.trade_entry_bar is not None else 0)
                dt = self.data.index[-1]
                entry_price = self.order_entry_price if self.order_entry_price is not None else 0
                exit_price = trade.price
                profit_usd = trade.pnl
                direction = self.entry_signal
                if direction == 'long':
                    profit_pips = (exit_price - entry_price) / self.params.pip_cost
                    intra_dd = (entry_price - self.trade_low) / self.params.pip_cost if self.trade_low is not None else 0
                elif direction == 'short':
                    profit_pips = (entry_price - exit_price) / self.params.pip_cost
                    intra_dd = (self.trade_high - entry_price) / self.params.pip_cost if self.trade_high is not None else 0
                else:
                    profit_pips = 0
                    intra_dd = 0
                print(f"[DEBUG TRADE ENTRY] Signal: {self.entry_signal} | Entry Profit (pips): {self.entry_profit:.2f} | "
                    f"Entry Risk (pips): {self.entry_risk:.2f} | Entry RR: {self.entry_rr:.2f}", flush=True)
                print(f"[DEBUG TRADE CLOSED] ({direction}): Date={dt}, Entry={entry_price:.5f}, Exit={exit_price:.5f}, "
                    f"Profit (USD)={profit_usd:.2f}, Pips={profit_pips:.2f}, Duration={duration} bars, MaxDD={intra_dd:.2f}, "
                    f"Balance={self.broker.equity:.2f}", flush=True)
                # Reset trade-specific variables.
                self.order_entry_price = None
                self.current_tp = None
                self.current_sl = None
                self.entry_signal = None

    # -------------------------------------------------------------------------
    # Dummy methods for interface compatibility
    # -------------------------------------------------------------------------
    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

    def build_model(self, input_shape):
        print("build_model() not applicable for trading strategy plugin.")

    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val=None, y_val=None):
        print("train() not applicable for trading strategy plugin.")

    def predict(self, data):
        print("predict() not applicable for trading strategy plugin.")
        return None

    def calculate_mse(self, y_true, y_pred):
        print("calculate_mse() not applicable for trading strategy plugin.")
        return None

    def calculate_mae(self, y_true, y_pred):
        print("calculate_mae() not applicable for trading strategy plugin.")
        return None

    def save(self, file_path):
        print("save() not applicable for trading strategy plugin.")

    def load(self, file_path):
        print("load() not applicable for trading strategy plugin.")
