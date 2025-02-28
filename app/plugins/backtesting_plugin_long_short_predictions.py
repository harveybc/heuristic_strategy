import os
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from datetime import datetime


class Plugin:
    """
    Plugin for Heuristic Trading Strategy using backtesting.py

    La estrategia se basa en el análisis de predicciones a corto y largo plazo para calcular
    el beneficio potencial y riesgo (en pips) de operar LONG o SHORT, comparar la tasa beneficio/riesgo
    (RR) y ejecutar la orden en la dirección con mejor RR (si se supera un umbral). Además, se monitorea
    cada tick (hora) para cerrar la posición si se alcanza el Take Profit (TP) o si las predicciones indican
    que el Stop Loss (SL) se alcanzará antes del TP.
    """

    # Parámetros por defecto (optimizables)
    plugin_params = {
        'pip_cost': 0.00001,
        'rel_volume': 0.02,
        'min_order_volume': 10000,
        'max_order_volume': 1000000,
        'leverage': 1000,
        'profit_threshold': 5,       # pips
        'min_drawdown_pips': 10,
        'tp_multiplier': 0.9,
        'sl_multiplier': 2.0,
        'lower_rr_threshold': 0.5,
        'upper_rr_threshold': 2.0,
        'max_trades_per_5days': 3,
        'time_horizon': 3            # en días (cada día = 24 ticks)
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
        import os
        import pandas as pd
        from backtesting import Backtest

        # Desempaquetado: si el candidato tiene 12 valores se usan todos; si tiene 6, se usan los actuales para los demás.
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

        # Actualizar parámetros del plugin
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

        # Auto-generar predicciones si no se han proporcionado
        if (config.get('hourly_predictions_file') is None) and (config.get('daily_predictions_file') is None):
            print(f"[evaluate_candidate] Auto-generating predictions using time_horizon={int(time_horizon)} for candidate {individual}.")
            from data_processor import process_data
            processed = process_data(config)
            hourly_predictions = processed["hourly"]
            daily_predictions = processed["daily"]

        # Fusionar predicciones (opcional, para preservar la interfaz)
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
        # Guardar temporalmente las predicciones (opcional)
        temp_pred_file = "temp_predictions.csv"
        merged_df.reset_index().to_csv(temp_pred_file, index=False)

        # Preparar la "extra" información para la estrategia:
        extra = {"hourly": hourly_predictions, "daily": daily_predictions, "params_config": self.params}

        # Asegurarse de que base_data sea un DataFrame y adjuntar extra como atributo.
        base_data = base_data.copy()
        if not isinstance(base_data.index, pd.DatetimeIndex):
            base_data.index = pd.to_datetime(base_data.index)
        base_data.extra = extra  # <--- Aquí se guarda la información extra

        # Crear el objeto Backtest sin pasar "extra" como argumento, pues se adjunta al DataFrame.
        from backtesting import Backtest
        bt_sim = Backtest(base_data, self.HeuristicStrategy, cash=10000, commission=0.0, exclusive_orders=True)
        perf = bt_sim.run()
        final_balance = perf["Equity"].iloc[-1]
        profit = final_balance - 10000.0
        print(f"[ZIPLINE ANALYZE] Final Balance: {final_balance:.2f} | Profit: {profit:.2f}", flush=True)
        # Retornar profit y performance
        return (profit, {"portfolio": perf})

    class HeuristicStrategy(Strategy):
        def init(self):
            # Recuperar la información extra del DataFrame
            self.hourly_preds = self.data.extra["hourly"]
            self.daily_preds = self.data.extra["daily"]
            self.params_config = self.data.extra["params_config"]
            # Variables de control
            self.order_active = False
            self.current_signal = None
            self.TP = None
            self.SL = None
            self.last_trade_date = None
            self.trade_low = None
            self.trade_high = None
            self.entry_metrics = {}  # Para debug: almacenar métricas de entrada

        def next(self):
            current_dt = self.data.index[-1].to_pydatetime()
            current_price = self.data.Close[-1]
            print(f"[ZIPLINE DEBUG TICK] Date: {current_dt} | Price: {current_price:.5f}", flush=True)

            # Si hay posición abierta, evaluar condiciones de salida.
            if self.order_active:
                if self.current_signal == "long":
                    # Para salida, se usan predicciones (falla: se usa la fila correspondiente o la última disponible)
                    if current_dt in self.daily_preds.index:
                        row_daily = self.daily_preds.loc[current_dt]
                    else:
                        row_daily = self.daily_preds.iloc[-1]
                    if current_dt in self.hourly_preds.index:
                        row_hourly = self.hourly_preds.loc[current_dt]
                    else:
                        row_hourly = self.hourly_preds.iloc[-1]
                    preds_combined = list(row_hourly.values) + list(row_daily.values)
                    predicted_min = min(preds_combined)
                    print(f"[ZIPLINE DEBUG EXIT - LONG] Price: {current_price:.5f} | TP: {self.TP:.5f} | SL: {self.SL:.5f} | Predicted Min: {predicted_min:.5f}", flush=True)
                    if current_price >= self.TP or predicted_min < self.SL:
                        self.position.close()
                        self.order_active = False
                        print("[ZIPLINE DEBUG EXIT - LONG] Exiting long position.", flush=True)
                elif self.current_signal == "short":
                    if current_dt in self.hourly_preds.index:
                        row_hourly = self.hourly_preds.loc[current_dt]
                    else:
                        row_hourly = self.hourly_preds.iloc[-1]
                    if current_dt in self.daily_preds.index:
                        row_daily = self.daily_preds.loc[current_dt]
                    else:
                        row_daily = self.daily_preds.iloc[-1]
                    preds_combined = list(row_hourly.values) + list(row_daily.values)
                    predicted_max = max(preds_combined)
                    print(f"[ZIPLINE DEBUG EXIT - SHORT] Price: {current_price:.5f} | TP: {self.TP:.5f} | SL: {self.SL:.5f} | Predicted Max: {predicted_max:.5f}", flush=True)
                    if current_price <= self.TP or predicted_max > self.SL:
                        self.position.close()
                        self.order_active = False
                        print("[ZIPLINE DEBUG EXIT - SHORT] Exiting short position.", flush=True)
                return

            # Control de frecuencia: si la última operación fue hace menos de max_trades_per_5days, no entrar.
            if self.last_trade_date is not None:
                delta_days = (current_dt - self.last_trade_date).days
                if delta_days < self.params_config["max_trades_per_5days"]:
                    print("[ZIPLINE DEBUG ENTRY] Frequency limit active. Skipping entry.", flush=True)
                    return

            # Para entrada: usar predicciones diarias (fila correspondiente o última)
            if current_dt in self.daily_preds.index:
                row = self.daily_preds.loc[current_dt]
            else:
                row = self.daily_preds.iloc[-1]
            daily_vals = list(row.values)
            if not daily_vals or all(pd.isna(daily_vals)):
                return
            max_pred = max(daily_vals)
            min_pred = min(daily_vals)
            # Cálculos para LONG:
            ideal_profit_long = (max_pred - current_price) / self.params_config["pip_cost"]
            ideal_drawdown_long = max((current_price - min_pred) / self.params_config["pip_cost"],
                                      self.params_config["min_drawdown_pips"])
            rr_long = ideal_profit_long / ideal_drawdown_long if ideal_drawdown_long > 0 else 0
            TP_long = current_price + self.params_config["tp_multiplier"] * ideal_profit_long * self.params_config["pip_cost"]
            SL_long = current_price - self.params_config["sl_multiplier"] * ideal_drawdown_long * self.params_config["pip_cost"]

            # Cálculos para SHORT:
            ideal_profit_short = (current_price - min_pred) / self.params_config["pip_cost"]
            ideal_drawdown_short = max((max_pred - current_price) / self.params_config["pip_cost"],
                                       self.params_config["min_drawdown_pips"])
            rr_short = ideal_profit_short / ideal_drawdown_short if ideal_drawdown_short > 0 else 0
            TP_short = current_price - self.params_config["tp_multiplier"] * ideal_profit_short * self.params_config["pip_cost"]
            SL_short = current_price + self.params_config["sl_multiplier"] * ideal_drawdown_short * self.params_config["pip_cost"]

            print(f"[ZIPLINE DEBUG ENTRY] Price: {current_price:.5f}", flush=True)
            print(f"[ZIPLINE DEBUG ENTRY - LONG] Profit: {ideal_profit_long:.2f} pips, Risk: {ideal_drawdown_long:.2f} pips, RR: {rr_long:.2f}, TP: {TP_long:.5f}, SL: {SL_long:.5f}", flush=True)
            print(f"[ZIPLINE DEBUG ENTRY - SHORT] Profit: {ideal_profit_short:.2f} pips, Risk: {ideal_drawdown_short:.2f} pips, RR: {rr_short:.2f}, TP: {TP_short:.5f}, SL: {SL_short:.5f}", flush=True)

            if ideal_profit_long < self.params_config["profit_threshold"] and ideal_profit_short < self.params_config["profit_threshold"]:
                print("[ZIPLINE DEBUG ENTRY] Profit threshold not met for either side. Skipping entry.", flush=True)
                return

            if rr_long >= rr_short:
                signal = "long"
                chosen_TP = TP_long
                chosen_SL = SL_long
                chosen_RR = rr_long
            else:
                signal = "short"
                chosen_TP = TP_short
                chosen_SL = SL_short
                chosen_RR = rr_short

            # Guardar métricas de entrada para debug en notify_trade
            self.entry_metrics = {
                "signal": signal,
                "entry_profit": ideal_profit_long if signal=="long" else ideal_profit_short,
                "entry_risk": ideal_drawdown_long if signal=="long" else ideal_drawdown_short,
                "entry_rr": chosen_RR
            }

            # Ejecutar la orden. Con backtesting.py se simula operar con el 100% del capital.
            if signal == "long":
                self.buy()
                print("[ZIPLINE DEBUG ENTRY] LONG order executed.", flush=True)
            else:
                self.sell()
                print("[ZIPLINE DEBUG ENTRY] SHORT order executed.", flush=True)

            self.TP = chosen_TP
            self.SL = chosen_SL
            self.order_active = True
            self.last_trade_date = current_dt

        def notify_trade(self, trade):
            if trade.isclosed:
                duration = len(self.data) - self._tradelength if hasattr(self, '_tradelength') else 0
                current_balance = self.data.df["Equity"].iloc[-1] if "Equity" in self.data.df.columns else None
                print(f"[ZIPLINE DEBUG TRADE] Trade closed. PnL: {trade.pnl:.2f} | Duration: {duration} bars", flush=True)
                # Resetear la señal
                self.order_active = False



    # Métodos dummy para la interfaz
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
