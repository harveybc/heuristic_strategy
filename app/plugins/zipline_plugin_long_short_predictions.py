import os
import pandas as pd
import numpy as np
from zipline.api import (
    order_target_percent, record, symbol, set_commission, set_slippage
)
from zipline import run_algorithm
from datetime import datetime

class Plugin:
    """
    Plugin for Heuristic Trading Strategy using Zipline.
    
    La estrategia se compone de:
      1. APERTURA:
         - Se analizan las predicciones a corto y largo plazo para obtener:
             • El valor máximo proyectado (potencial ganancia) y el valor mínimo proyectado (potencial riesgo) 
               (así como los extremos previos a dichos valores, en la práctica se usan las predicciones diarias).
         - Se calcula el beneficio potencial (en pips) y el riesgo (en pips) para una orden LONG y para una SHORT.
         - Se calcula la tasa beneficio/riesgo (RR) y se compara entre ambas señales.
         - Si el beneficio potencial (de alguna de las dos) supera un umbral mínimo configurable y la RR es la
           mejor, se abre la posición en esa dirección; además se definen los niveles de Take Profit (TP) y Stop Loss (SL)
           usando multiplicadores configurables y los extremos obtenidos.
      
      2. CIERRE:
         - Una vez abierta la posición, se monitorea en cada tick (hora) el precio y se consultan ambas predicciones
           (usando “fallback” si la fecha no coincide exactamente) para evaluar:
             • Si el precio actual alcanza TP, se cierra la posición.
             • Si la predicción (corto y/o largo) indica que el precio se moverá de forma adversa (alcanzando el SL)
               antes de que se cumpla el TP estimado, se cierra la posición de forma anticipada.
      
    Los parámetros son configurables y optimizables.
    """
    # Parámetros por defecto (optimizables)
    plugin_params = {
        'pip_cost': 0.00001,
        'rel_volume': 0.02,          # Fracción del cash disponible
        'min_order_volume': 10000,
        'max_order_volume': 1000000,
        'leverage': 1000,
        'profit_threshold': 5,       # Umbral mínimo de beneficio potencial (pips)
        'min_drawdown_pips': 10,
        'tp_multiplier': 0.9,
        'sl_multiplier': 2.0,
        'lower_rr_threshold': 0.5,
        'upper_rr_threshold': 2.0,
        'max_trades_per_5days': 3,
        'time_horizon': 3            # Número de días para estimar TP (cada día equivale a 24 ticks)
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
        Evalúa un candidato (parámetros) usando los datasets (base, predicciones horarias y diarias).
        Si no se proporcionan archivos de predicción, se auto-generan.
        Devuelve el profit final y algunos datos de performance.
        """
        import os
        import pandas as pd

        # Desempaquetado: si individual tiene 12 valores se usan todos; si tiene 6 se asumen optimizados
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

        # Actualizar parámetros
        self.params['pip_cost'] = pip_cost
        self.params['rel_volume'] = rel_volume
        self.params['min_order_volume'] = min_order_volume
        self.params['max_order_volume'] = max_order_volume
        self.params['leverage'] = leverage
        self.params['profit_threshold'] = profit_threshold
        self.params['min_drawdown_pips'] = min_drawdown_pips
        self.params['tp_multiplier'] = tp_multiplier
        self.params['sl_multiplier'] = sl_multiplier
        self.params['lower_rr_threshold'] = lower_rr
        self.params['upper_rr_threshold'] = upper_rr
        self.params['time_horizon'] = int(time_horizon)

        # Auto-generar predicciones si no se han proporcionado
        if (config.get('hourly_predictions_file') is None) and (config.get('daily_predictions_file') is None):
            print(f"[evaluate_candidate] Auto-generating predictions using time_horizon={int(time_horizon)} for candidate {individual}.")
            config["time_horizon"] = int(time_horizon)
            from data_processor import process_data
            processed = process_data(config)
            hourly_predictions = processed["hourly"]
            daily_predictions = processed["daily"]

        # Fusionar predicciones (se espera que tengan índice datetime)
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
        if merged_df.index.name is None or merged_df.index.name != "DATE_TIME":
            merged_df.index.name = "DATE_TIME"
        temp_pred_file = "temp_predictions.csv"
        merged_df.reset_index().to_csv(temp_pred_file, index=False)

        # Definir fechas de inicio y fin de la simulación a partir de base_data.
        start_date = base_data.index.min().to_pydatetime()
        end_date = base_data.index.max().to_pydatetime()

        # Función initialize para Zipline.
        def initialize(context):
            context.asset = symbol("A")  # Símbolo ficticio
            context.params = self.params.copy()
            context.hourly_preds = hourly_predictions
            context.daily_preds = daily_predictions
            context.order_active = False
            context.current_signal = None
            context.TP = None
            context.SL = None
            context.entry_profit = None
            context.entry_risk = None
            context.entry_rr = None
            context.last_trade_date = None
            set_commission(commission=0.0)
            set_slippage(slippage=0.0)
            print(f"[ZIPLINE INIT] Simulation from {start_date} to {end_date}.", flush=True)

        # Función handle_data que se ejecuta cada tick (hora)
        def handle_data(context, data):
            current_dt = data.current_dt
            current_price = data.current(context.asset, "price")
            print(f"[ZIPLINE DEBUG TICK] Date: {current_dt} | Price: {current_price:.5f}", flush=True)
            current_balance = data.portfolio.value

            # Si hay posición activa, evaluar condiciones de salida
            if context.order_active:
                # Obtener predicciones para current_dt; si no existen, usar la última disponible.
                if current_dt in context.daily_preds.index:
                    row_daily = context.daily_preds.loc[current_dt]
                else:
                    row_daily = context.daily_preds.iloc[-1]
                if current_dt in context.hourly_preds.index:
                    row_hourly = context.hourly_preds.loc[current_dt]
                else:
                    row_hourly = context.hourly_preds.iloc[-1]
                preds_hourly = row_hourly.values.tolist()
                preds_daily = row_daily.values.tolist()
                combined_vals = preds_hourly + preds_daily

                if context.current_signal == "long":
                    predicted_min = min(combined_vals)
                    print(f"[ZIPLINE DEBUG EXIT - LONG] Price: {current_price:.5f} | TP: {context.TP:.5f} | SL: {context.SL:.5f} | Predicted Min: {predicted_min:.5f}", flush=True)
                    if current_price >= context.TP or predicted_min < context.SL:
                        order_target_percent(context.asset, 0)
                        context.order_active = False
                        print("[ZIPLINE DEBUG EXIT - LONG] Exiting long position.", flush=True)
                        return
                elif context.current_signal == "short":
                    predicted_max = max(combined_vals)
                    print(f"[ZIPLINE DEBUG EXIT - SHORT] Price: {current_price:.5f} | TP: {context.TP:.5f} | SL: {context.SL:.5f} | Predicted Max: {predicted_max:.5f}", flush=True)
                    if current_price <= context.TP or predicted_max > context.SL:
                        order_target_percent(context.asset, 0)
                        context.order_active = False
                        print("[ZIPLINE DEBUG EXIT - SHORT] Exiting short position.", flush=True)
                        return
                return  # Mientras haya posición, no se buscan nuevas entradas.

            # Si no hay posición, controlar frecuencia de trading.
            if context.last_trade_date is not None:
                if (current_dt - context.last_trade_date).days < 5:
                    print("[ZIPLINE DEBUG ENTRY] Frequency limit active. Skipping entry.", flush=True)
                    return

            # Obtener predicciones para la entrada (usamos daily_preds; fallback a la última fila).
            if current_dt in context.daily_preds.index:
                row = context.daily_preds.loc[current_dt]
            else:
                row = context.daily_preds.iloc[-1]
            daily_vals = row.values.tolist()
            if not daily_vals or all(pd.isna(daily_vals)):
                return
            max_pred = max(daily_vals)
            min_pred = min(daily_vals)

            # Cálculos para orden LONG.
            ideal_profit_long = (max_pred - current_price) / context.params["pip_cost"]
            ideal_drawdown_long = max((current_price - min_pred) / context.params["pip_cost"], context.params["min_drawdown_pips"])
            rr_long = ideal_profit_long / ideal_drawdown_long if ideal_drawdown_long > 0 else 0
            TP_long = current_price + context.params["tp_multiplier"] * ideal_profit_long * context.params["pip_cost"]
            SL_long = current_price - context.params["sl_multiplier"] * ideal_drawdown_long * context.params["pip_cost"]

            # Cálculos para orden SHORT.
            ideal_profit_short = (current_price - min_pred) / context.params["pip_cost"]
            ideal_drawdown_short = max((max_pred - current_price) / context.params["pip_cost"], context.params["min_drawdown_pips"])
            rr_short = ideal_profit_short / ideal_drawdown_short if ideal_drawdown_short > 0 else 0
            TP_short = current_price - context.params["tp_multiplier"] * ideal_profit_short * context.params["pip_cost"]
            SL_short = current_price + context.params["sl_multiplier"] * ideal_drawdown_short * context.params["pip_cost"]

            print(f"[ZIPLINE DEBUG ENTRY] Price: {current_price:.5f}", flush=True)
            print(f"[ZIPLINE DEBUG ENTRY - LONG] Profit: {ideal_profit_long:.2f} pips, Risk: {ideal_drawdown_long:.2f} pips, RR: {rr_long:.2f}, TP: {TP_long:.5f}, SL: {SL_long:.5f}", flush=True)
            print(f"[ZIPLINE DEBUG ENTRY - SHORT] Profit: {ideal_profit_short:.2f} pips, Risk: {ideal_drawdown_short:.2f} pips, RR: {rr_short:.2f}, TP: {TP_short:.5f}, SL: {SL_short:.5f}", flush=True)

            if ideal_profit_long < context.params["profit_threshold"] and ideal_profit_short < context.params["profit_threshold"]:
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

            context.entry_profit = ideal_profit_long if signal == "long" else ideal_profit_short
            context.entry_risk = ideal_drawdown_long if signal == "long" else ideal_drawdown_short
            context.entry_rr = chosen_RR
            context.current_signal = signal

            # Cálculo del tamaño de la orden basado en el cash disponible.
            current_cash = data.portfolio.cash
            order_size = min(current_cash * context.params["rel_volume"] * context.params["leverage"], context.params["max_order_volume"])
            order_size = max(order_size, context.params["min_order_volume"])

            if signal == "long":
                order_target_percent(context.asset, 1.0)
                print(f"[ZIPLINE DEBUG ENTRY] LONG order executed. (Approx Order Size: {order_size})", flush=True)
            else:
                order_target_percent(context.asset, -1.0)
                print(f"[ZIPLINE DEBUG ENTRY] SHORT order executed. (Approx Order Size: {order_size})", flush=True)

            context.TP = chosen_TP
            context.SL = chosen_SL
            context.order_active = True
            context.last_trade_date = current_dt
            print(f"[ZIPLINE DEBUG ENTRY] Signal: {signal} | Entry Profit: {context.entry_profit:.2f} pips, Entry Risk: {context.entry_risk:.2f} pips, RR: {context.entry_rr:.2f}", flush=True)

        def analyze(perf):
            final_balance = perf['portfolio_value'].iloc[-1]
            profit = final_balance - 10000.0
            print(f"[ZIPLINE ANALYZE] Final Balance: {final_balance:.2f} | Profit: {profit:.2f}", flush=True)

        perf = run_algorithm(
            start=start_date,
            end=end_date,
            initialize=initialize,
            handle_data=handle_data,
            capital_base=10000,
            data_frequency='hourly'
        )
        final_balance = perf['portfolio_value'].iloc[-1]
        profit = final_balance - 10000.0
        print(f"Evaluated candidate {individual} -> Profit: {profit:.2f}", flush=True)
        return (profit, {"portfolio": perf})

    # Métodos dummy para compatibilidad.
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
