import datetime
import os
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy

# Global variable to hold extra info for the strategy.
EXTRA_INFO = None

class Plugin:
    """
    Plugin for Heuristic Trading Strategy using backtesting.py.

    - Implements a heuristic trading strategy based on short-term (hourly) and long-term (daily) predictions.
    - The strategy compares profit/risk ratios for long and short entries and only enters if the profit potential exceeds a threshold.
    - Exit conditions are based on reaching take profit or stop loss.
    - The strategy operates one order at a time.
    - Parameters are configurable and optimizable.
    """
    # Default plugin parameters (must be present for optimizer integration)
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
        'time_horizon': 3           # In days (used for expected TP tick, e.g. 3 days = 72 ticks)
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
        # If candidate has 12 values, use all; if 6, then use defaults for the others.
        if len(individual) == 12:
            (pip_cost, rel_volume, min_order_volume, max_order_volume, leverage, profit_threshold,
             min_drawdown_pips, tp_multiplier, sl_multiplier, lower_rr, upper_rr, time_horizon) = individual
        elif len(individual) == 6:
            (profit_threshold, tp_multiplier, sl_multiplier, lower_rr, upper_rr, time_horizon) = individual
            pip_cost = self.params['pip_cost']
            rel_volume = self.params['rel_volume']
            min_order_volume = self.params['min_order_volume']
            max_order_volume = self.params['max_order_volume']
            leverage = self.params['leverage']
            min_drawdown_pips = self.params['min_drawdown_pips']
        else:
            raise ValueError(f"Expected candidate with 6 or 12 values, got {len(individual)}")

        # Update plugin parameters.
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

        # Auto-generate predictions if not provided.
        if (config.get('hourly_predictions_file') is None) and (config.get('daily_predictions_file') is None):
            print(f"[evaluate_candidate] Auto-generating predictions using time_horizon={int(time_horizon)} for candidate {individual}.")
            from data_processor import process_data
            processed = process_data(config)
            hourly_predictions = processed["hourly"]
            daily_predictions = processed["daily"]

        # Merge predictions.
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

        # Normalize base_data columns for backtesting.py.
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
        if not isinstance(base_data.index, pd.DatetimeIndex):
            base_data.index = pd.to_datetime(base_data.index)
        base_data = base_data.copy()

        # Instead of attaching extra info to base_data, store it in a global variable.
        global EXTRA_INFO
        extra = {"hourly": hourly_predictions, "daily": daily_predictions, "params_config": self.params}
        EXTRA_INFO = extra

        # Create and run the Backtest.
        bt_sim = Backtest(base_data, self.HeuristicStrategy, cash=10000, commission=0.0, exclusive_orders=True)
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
          - Computes potential profit and risk for long and short orders using short- and long-term predictions.
          - Compares profit/risk (RR) for long vs. short; if the potential profit exceeds a threshold,
            enters the trade (long if RR_long >= RR_short, else short).
          - Order volume is computed proportionally between min_order_volume and max_order_volume.
        Exit:
          - At each tick (hour), if a position is open, the strategy checks if:
              * For long: the current price has reached TP or fallen below SL.
              * For short: the current price has reached TP or risen above SL.
        """
        def __init__(self, data, **kwargs):
            super().__init__(data, **kwargs)
            # Instead of reading extra from data (which is not allowed), get it from global EXTRA_INFO.
            global EXTRA_INFO
            self.extra = EXTRA_INFO
            self.trade_entry_bar = None
            self.trade_low = None
            self.trade_high = None
            self.entry_profit = None
            self.entry_risk = None
            self.entry_rr = None
            self.entry_signal = None
            self.order_entry_price = None
            self.current_tp = None
            self.current_sl = None
            self.current_direction = None

        def next(self):
            current_price = self.data.Close[-1]
            dt = self.data.index[-1]

            # If a position is open, check exit conditions.
            if self.position:
                if self.entry_signal == 'long':
                    if (self.trade_low is None) or (current_price < self.trade_low):
                        self.trade_low = current_price
                    print(f"[DEBUG EXIT - LONG] {dt} | Current Price: {current_price:.5f} | TP: {self.current_tp:.5f} | SL: {self.current_sl:.5f}", flush=True)
                    if current_price >= self.current_tp:
                        print("[DEBUG EXIT - LONG] TP reached. Closing position.", flush=True)
                        self.position.close()
                        return
                    if current_price < self.current_sl:
                        print("[DEBUG EXIT - LONG] Price below SL. Closing position early.", flush=True)
                        self.position.close()
                        return
                elif self.entry_signal == 'short':
                    if (self.trade_high is None) or (current_price > self.trade_high):
                        self.trade_high = current_price
                    print(f"[DEBUG EXIT - SHORT] {dt} | Current Price: {current_price:.5f} | TP: {self.current_tp:.5f} | SL: {self.current_sl:.5f}", flush=True)
                    if current_price <= self.current_tp:
                        print("[DEBUG EXIT - SHORT] TP reached. Closing position.", flush=True)
                        self.position.close()
                        return
                    if current_price > self.current_sl:
                        print("[DEBUG EXIT - SHORT] Price above SL. Closing position early.", flush=True)
                        self.position.close()
                        return
                return  # Do not check for new entries if a position is open.
            # No open position: update extremes.
            self.trade_low = current_price
            self.trade_high = current_price

            # Retrieve daily predictions from extra info.
            daily_preds_df = self.extra["daily"]
            try:
                row = daily_preds_df.loc[dt]
            except KeyError:
                row = daily_preds_df.iloc[-1]
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

            # Determine which signal to take.
            if (ideal_profit_pips_buy >= self.params.profit_threshold) and (rr_buy >= rr_sell):
                signal = 'long'
                self.current_tp = tp_buy
                self.current_sl = sl_buy
            elif (ideal_profit_pips_sell >= self.params.profit_threshold) and (rr_sell > rr_buy):
                signal = 'short'
                self.current_tp = tp_sell
                self.current_sl = sl_sell
            else:
                print("[DEBUG ENTRY] Profit threshold condition not met for either signal.", flush=True)
                return

            # Store entry metrics for later debugging.
            self.entry_profit = ideal_profit_pips_buy if signal == 'long' else ideal_profit_pips_sell
            self.entry_risk = ideal_drawdown_pips_buy if signal == 'long' else ideal_drawdown_pips_sell
            self.entry_rr = rr_buy if signal == 'long' else rr_sell
            self.entry_signal = signal

            # Compute order size.
            order_size = self.compute_size(self.entry_rr)
            if order_size <= 0:
                print("[DEBUG] Order size <= 0, skipping trade", flush=True)
                return

            self.trade_entry_dates.append(dt)
            self.trade_entry_bar = len(self)
            self.current_volume = order_size

            if signal == 'long':
                self.buy(size=order_size)
                self.current_direction = 'long'
                print(f"[DEBUG ENTRY] LONG order executed. Order Size: {order_size}", flush=True)
            elif signal == 'short':
                self.sell(size=order_size)
                self.current_direction = 'short'
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
                self.order_entry_price = None
                self.current_tp = None
                self.current_sl = None
                self.current_direction = None
                self.entry_signal = None

        def stop(self):
            n_trades = 0
            if hasattr(self, 'trades') and self.trades:
                n_trades = len(self.trades)
            if n_trades > 0:
                avg_profit_usd = sum(t['pnl'] for t in self.trades) / n_trades
                avg_profit_pips = sum(t['pips'] for t in self.trades) / n_trades
                avg_duration = sum(t['duration'] for t in self.trades) / n_trades
                avg_max_dd = sum(t['max_dd'] for t in self.trades) / n_trades
            else:
                avg_profit_usd = avg_profit_pips = avg_duration = avg_max_dd = 0
            final_balance = self.broker.equity
            print("\n==== Summary ====")
            print(f"Initial Balance (USD): {self.initial_balance:.2f}")
            print(f"Final Balance (USD):   {final_balance:.2f}")
            print(f"Number of Trades: {n_trades}")
            print(f"Average Profit (USD): {avg_profit_usd:.2f}")
            print(f"Average Profit (pips): {avg_profit_pips:.2f}")
            print(f"Average Max Drawdown (pips): {avg_max_dd:.2f}")
            print(f"Average Trade Duration (bars): {avg_duration:.2f}")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(self.date_history, self.balance_history, label="Balance")
            plt.xlabel("Date")
            plt.ylabel("Balance (USD)")
            plt.title("Balance vs Date")
            plt.legend()
            plt.savefig("balance_plot.png")
            plt.close()

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
