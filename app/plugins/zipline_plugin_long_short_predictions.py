import pandas as pd
import numpy as np
import json
import os
import zipline
from zipline.api import order_target_percent, record, symbol, set_slippage, set_commission, schedule_function, date_rules, time_rules
from zipline import run_algorithm
from datetime import datetime, timedelta

class Plugin:
    """
    Plugin for Heuristic Trading Strategy using Zipline.
    
    This plugin preserves the interface of the Backtrader version:
      - It defines plugin_params (configurable/optimizable parameters).
      - It provides set_params, get_debug_info, get_optimizable_params.
      - The evaluate_candidate method runs the Zipline algorithm using the provided datasets (base_data,
        hourly_predictions, daily_predictions) and returns performance metrics.
    
    La estrategia se basa en:
      1. Evaluar predicciones a corto plazo (6 valores para las próximas 6 horas) y a largo plazo (6 valores para los siguientes 6 días).
      2. Para cada tick (hora) se calcula el beneficio potencial y riesgo para operaciones LONG y SHORT usando los extremos de las predicciones a largo plazo.
      3. Se selecciona la señal (LONG o SHORT) que ofrezca la mejor tasa beneficio/riesgo (si supera un umbral mínimo).
      4. Una vez abierta la posición, se monitorean en cada tick condiciones de salida: se cierra si se alcanza TP o SL,
         o si, tras un horizonte de tiempo (time_horizon convertido a ticks) y las predicciones indican que TP no se alcanzará, se cierra la operación.
      5. Sólo se mantiene una posición a la vez, y se controla la frecuencia máxima de operaciones en 5 días.
    """

    # Default plugin parameters
    plugin_params = {
        'pip_cost': 0.00001,
        'rel_volume': 0.02,
        'min_order_volume': 10000,
        'max_order_volume': 1000000,
        'leverage': 1000,
        'profit_threshold': 5,
        'min_drawdown_pips': 10,
        'tp_multiplier': 0.9,
        'sl_multiplier': 2.0,
        'lower_rr_threshold': 0.5,
        'upper_rr_threshold': 2.0,
        'max_trades_per_5days': 3,
        'time_horizon': 3  # en días (por defecto)
    }

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.trades = []

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def get_debug_info(self):
        return {key: self.params[key] for key in self.params}

    def get_optimizable_params(self):
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
        """
        Evaluates a candidate parameter set using the provided datasets.
        The base_data, hourly_predictions, and daily_predictions should be aligned DataFrames with a DatetimeIndex.
        """
        # Unpack candidate parameters
        profit_threshold, tp_multiplier, sl_multiplier, lower_rr, upper_rr, time_horizon = individual
        # Update our parameters
        self.params["profit_threshold"] = profit_threshold
        self.params["tp_multiplier"] = tp_multiplier
        self.params["sl_multiplier"] = sl_multiplier
        self.params["lower_rr_threshold"] = lower_rr
        self.params["upper_rr_threshold"] = upper_rr
        self.params["time_horizon"] = int(time_horizon)

        # If prediction DataFrames are empty, try to auto-generate them using data_processor logic.
        if hourly_predictions is None or hourly_predictions.empty or daily_predictions is None or daily_predictions.empty:
            from app.data_processor import process_data
            datasets = process_data(config)
            hourly_predictions = datasets["hourly"]
            daily_predictions = datasets["daily"]
            base_data = datasets["base"]

        # Merge predictions (optional step: here we assume they are already aligned as needed)
        # For Zipline, we'll pass them to context via a dictionary.
        predictions = {
            "hourly": hourly_predictions,
            "daily": daily_predictions
        }

        # Define the start and end dates for the simulation from base_data index
        start_date = base_data.index.min().to_pydatetime()
        end_date = base_data.index.max().to_pydatetime()

        # Define the algorithm using nested functions.
        def initialize(context):
            context.asset = symbol("A")  # Se usará un símbolo ficticio "A"
            context.params = self.params.copy()
            # Save predictions in context (they are DataFrames with a datetime index)
            context.predictions = predictions
            # Set initial state
            context.order_active = False
            context.entry_price = None
            context.entry_tick = None
            context.last_trade_date = None
            # Configure commission and slippage (pueden ser parámetros fijos)
            set_commission(commission=0.0000)
            set_slippage(slippage=0.0000)
            # Schedule handle_data every hour
            schedule_function(handle_data, date_rules.every_day(), time_rules.every_hour())

        def handle_data(context, data):
            current_dt = data.current_dt
            current_price = data.current(context.asset, "price")
            # Debug: Print tick info
            print(f"[ZIPLINE DEBUG TICK] Date: {current_dt} | Current Price: {current_price:.5f}")
            # If an order is active, check exit conditions:
            if context.order_active:
                # Retrieve predictions for current_dt from hourly and daily predictions
                # (Assuming that the predictions DataFrames have a DatetimeIndex that can be reindexed with current_dt)
                try:
                    preds_hourly = context.predictions["hourly"].loc[current_dt]
                except KeyError:
                    preds_hourly = None
                try:
                    preds_daily = context.predictions["daily"].loc[current_dt]
                except KeyError:
                    preds_daily = None
                if preds_hourly is not None and preds_daily is not None:
                    combined_pred_min = min(preds_hourly.values.tolist() + preds_daily.values.tolist())
                    combined_pred_max = max(preds_hourly.values.tolist() + preds_daily.values.tolist())
                    long_term_pred_max = max(preds_daily.values.tolist())
                    long_term_pred_min = min(preds_daily.values.tolist())
                else:
                    combined_pred_min = current_price
                    combined_pred_max = current_price
                    long_term_pred_max = current_price
                    long_term_pred_min = current_price

                # Print debug of exit variables
                print(f"[ZIPLINE DEBUG EXIT] Current Price: {current_price:.5f} | TP: {context.TP:.5f} | SL: {context.SL:.5f} | "
                      f"Combined Pred Min: {combined_pred_min:.5f} | Combined Pred Max: {combined_pred_max:.5f} | "
                      f"LongTerm Pred Max: {long_term_pred_max:.5f} | LongTerm Pred Min: {long_term_pred_min:.5f}")
                
                # Check exit conditions based on position type
                if context.current_signal == "long":
                    if current_price >= context.TP:
                        order_target_percent(context.asset, 0)
                        context.order_active = False
                        print("[ZIPLINE DEBUG EXIT] LONG TP reached. Exiting position.")
                    elif combined_pred_min < context.SL or long_term_pred_max < context.TP:
                        order_target_percent(context.asset, 0)
                        context.order_active = False
                        print("[ZIPLINE DEBUG EXIT] LONG exit condition met. Exiting position early.")
                elif context.current_signal == "short":
                    if current_price <= context.TP:
                        order_target_percent(context.asset, 0)
                        context.order_active = False
                        print("[ZIPLINE DEBUG EXIT] SHORT TP reached. Exiting position.")
                    elif combined_pred_max > context.SL or long_term_pred_min > context.TP:
                        order_target_percent(context.asset, 0)
                        context.order_active = False
                        print("[ZIPLINE DEBUG EXIT] SHORT exit condition met. Exiting position early.")
                return  # Exit early if position is active

            # Entry Logic: Only if no order is active and frequency conditions are met.
            # (For frequency control, we can use context.last_trade_date)
            if context.last_trade_date is not None:
                if (current_dt - context.last_trade_date).days < 5:
                    # Skip entry if too soon.
                    print("[ZIPLINE DEBUG ENTRY] Trade frequency limit active. Skipping entry.")
                    return

            # Calculate extremes from daily predictions for current_dt
            try:
                daily_vals = context.predictions["daily"].loc[current_dt].values.tolist()
            except KeyError:
                print("[ZIPLINE DEBUG ENTRY] No daily prediction for current time. Skipping entry.")
                return
            max_pred = max(daily_vals)
            min_pred = min(daily_vals)

            # Calculate potential profit and risk for LONG:
            profit_long = (max_pred - current_price) / context.params["pip_cost"]
            risk_long = max((current_price - min_pred) / context.params["pip_cost"], context.params["min_drawdown_pips"])
            rr_long = profit_long / risk_long if risk_long > 0 else 0

            # Calculate potential profit and risk for SHORT:
            profit_short = (current_price - min_pred) / context.params["pip_cost"]
            risk_short = max((max_pred - current_price) / context.params["pip_cost"], context.params["min_drawdown_pips"])
            rr_short = profit_short / risk_short if risk_short > 0 else 0

            print(f"[ZIPLINE DEBUG ENTRY] Current Price: {current_price:.5f}")
            print(f"[ZIPLINE DEBUG ENTRY - LONG] Profit: {profit_long:.2f} pips, Risk: {risk_long:.2f} pips, RR: {rr_long:.2f}")
            print(f"[ZIPLINE DEBUG ENTRY - SHORT] Profit: {profit_short:.2f} pips, Risk: {risk_short:.2f} pips, RR: {rr_short:.2f}", flush=True)

            # Check if the profit potential exceeds the profit_threshold.
            if profit_long < context.params["profit_threshold"] and profit_short < context.params["profit_threshold"]:
                print("[ZIPLINE DEBUG ENTRY] Profit threshold not met for either signal. Skipping entry.", flush=True)
                return

            # Decide on the signal based on the higher RR.
            if rr_long >= rr_short:
                signal = "long"
                chosen_profit = profit_long
                chosen_risk = risk_long
            else:
                signal = "short"
                chosen_profit = profit_short
                chosen_risk = risk_short

            # Set TP and SL based on the chosen signal.
            if signal == "long":
                TP = current_price + context.params["tp_multiplier"] * chosen_profit * context.params["pip_cost"]
                SL = current_price - context.params["sl_multiplier"] * chosen_risk * context.params["pip_cost"]
            else:
                TP = current_price - context.params["tp_multiplier"] * chosen_profit * context.params["pip_cost"]
                SL = current_price + context.params["sl_multiplier"] * chosen_risk * context.params["pip_cost"]

            # Store entry metrics in context for later debugging.
            context.entry_profit = chosen_profit
            context.entry_risk = chosen_risk
            context.entry_rr = chosen_profit / chosen_risk if chosen_risk > 0 else 0
            context.current_signal = signal

            # For order sizing, a simple approach: use the relative volume of the current portfolio.
            current_cash = data.portfolio.cash
            # Compute order size using min_order_volume and max_order_volume constraints.
            # (This is a simplified version; in a real plugin, you might want to calculate based on risk.)
            order_size = max(min(current_cash * context.params["rel_volume"] * context.params["leverage"], context.params["max_order_volume"]),
                             context.params["min_order_volume"])

            # Place order: For simplicity, we assume full allocation in one direction.
            if signal == "long":
                order_target_percent(context.asset, 1.0)  # 100% allocation long
                print(f"[ZIPLINE DEBUG ENTRY] LONG order executed. Order Size (approx): {order_size}", flush=True)
            else:
                order_target_percent(context.asset, -1.0)  # 100% allocation short
                print(f"[ZIPLINE DEBUG ENTRY] SHORT order executed. Order Size (approx): {order_size}", flush=True)

            # Save TP, SL and trade entry tick (simulated by the current time in hours)
            context.TP = TP
            context.SL = SL
            context.trade_entry_tick = current_dt
            # Calculate expected TP tick based on time_horizon (days converted to hours)
            context.expected_tp_tick = current_dt + timedelta(hours=context.params["time_horizon"] * 24)
            context.order_active = True
            # Record the last trade date for frequency control.
            context.last_trade_date = current_dt

            print(f"[ZIPLINE DEBUG ENTRY] Signal: {signal} | Entry Profit: {chosen_profit:.2f} pips, Entry Risk: {chosen_risk:.2f} pips, RR: {context.entry_rr:.2f}", flush=True)

        # End of handle_data

        # Run the algorithm with our defined initialize and handle_data.
        perf = run_algorithm(start=start_date,
                             end=end_date,
                             initialize=initialize,
                             capital_base=10000,
                             handle_data=handle_data,
                             data_frequency='hourly',
                             bundle='quantopian-quandl')  # O usa el bundle que tengas configurado

        # After simulation, we can extract final balance and other metrics.
        final_balance = perf['portfolio_value'].iloc[-1]
        profit = final_balance - 10000
        # Additional metrics (win %, max drawdown, etc.) can be computed from perf as needed.
        # Here se simplifica y se devuelve un diccionario con el profit y performance DataFrame.
        print(f"Evaluated candidate {individual} -> Profit: {profit:.2f}")
        return (profit, {"portfolio": perf})

# End of Plugin class
if __name__ == "__main__":
    pass
