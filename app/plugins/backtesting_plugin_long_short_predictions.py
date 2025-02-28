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
        """
        Evaluates a candidate parameter set using the provided datasets.
        It updates the plugin parameters and runs a simulation using backtesting.py.
        """
        import os
        import pandas as pd
        from backtesting import Backtest

        # Desempaquetado: si individual tiene 12 valores se usan todos; si tiene 6 se usan los valores actuales para los no optimizados.
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

        # Auto-generar predicciones si no se han proporcionado.
        if (config.get('hourly_predictions_file') is None) and (config.get('daily_predictions_file') is None):
            print(f"[evaluate_candidate] Auto-generating predictions using time_horizon={int(time_horizon)} for candidate {individual}.")
            from data_processor import process_data
            processed = process_data(config)
            hourly_predictions = processed["hourly"]
            daily_predictions = processed["daily"]

        # Fusionar predicciones en un DataFrame (este archivo se usará internamente por la estrategia).
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

        # Normalizar columnas del DataFrame base para backtesting.py:
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

        # Asegurarse de que el índice sea DatetimeIndex.
        base_data = base_data.copy()
        if not isinstance(base_data.index, pd.DatetimeIndex):
            base_data.index = pd.to_datetime(base_data.index)

        # Adjuntar información extra al DataFrame base (usando setattr, ya que no se pueden agregar columnas arbitrarias)
        extra = {"hourly": hourly_predictions, "daily": daily_predictions, "params_config": self.params}
        setattr(base_data, "extra", extra)

        # Crear y correr la simulación con backtesting.py
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
          - Uses short-term (hourly) and long-term (daily) predictions to compute:
              • Profit potential for long: (max_daily - current_price) / pip_cost.
              • Risk for long: max( (current_price - min_daily) / pip_cost, min_drawdown_pips ).
              • Profit potential for short: (current_price - min_daily) / pip_cost.
              • Risk for short: max( (max_daily - current_price) / pip_cost, min_drawdown_pips ).
          - Compares profit/risk (RR) of long and short; enters la operación (long o short) con mejor RR siempre
            que el potencial de ganancia supere un umbral configurable.
          - El volumen de la orden se determina de forma proporcional, con límites mínimos y máximos.
        
        Exit:
          - Monitorea en cada tick (hora) el comportamiento del mercado.
          - Para una posición long: cierra si el precio alcanza TP o si las predicciones (corto y largo) indican que
            se ha alcanzado (o se aproxima a) SL.
          - Para una posición short: cierra si el precio alcanza TP o si las predicciones indican que se ha alcanzado (o se
            aproxima a) SL.
        """
        def init(self):
            # Inicializar variables de seguimiento
            self.trade_entry_date = None
            self.trade_entry_bar = None
            self.trade_low = None
            self.trade_high = None
            self.entry_profit = None
            self.entry_risk = None
            self.entry_rr = None
            self.entry_signal = None

        def next(self):
            # Se asume que self.data.Close, self.data.Open, etc. están disponibles.
            current_price = self.data.Close[-1]
            dt = self.data.index[-1]

            # Registrar balance (usando el valor de la cartera de backtesting.py)
            # Nota: backtesting.py almacena la serie "Equity" en perf; para debug se podría usar self.equity.
            # Aquí no se registra balance manualmente.

            # Si hay posición abierta, evaluar condiciones de salida:
            if self.position:
                if self.entry_signal == 'long':
                    # Actualizar el mínimo alcanzado durante la operación
                    if (self.trade_low is None) or (current_price < self.trade_low):
                        self.trade_low = current_price
                    # Si el precio alcanza el TP o el mínimo predicho (calculado a partir de la información extra)
                    # se cierra la posición.
                    if current_price >= self.current_tp:
                        self.position.close()
                        return
                    # Para simplificar, si el valor de "Close" es menor al SL calculado, se cierra.
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
                return  # Si hay posición abierta, no se buscan nuevas entradas.

            # Sin posición: reiniciar extremos.
            self.trade_low = current_price
            self.trade_high = current_price

            # Obtener las predicciones del DataFrame extra (almacenado en base_data.extra)
            # Se espera que extra tenga claves "hourly" y "daily"
            extra = self.data.extra  # Obtenido desde base_data.extra (adjunto con setattr)
            # Para este ejemplo, usamos únicamente las predicciones diarias para determinar extremos.
            # Se espera que las predicciones diarias sean un DataFrame indexado por DATE_TIME.
            # Buscamos la fila correspondiente a la fecha actual (o la más reciente).
            try:
                row = extra["daily"].loc[dt]
            except KeyError:
                # Si no se encuentra la fecha, usar la última fila disponible.
                row = extra["daily"].iloc[-1]
            # Convertir las predicciones en una lista de números.
            daily_preds = [row[col] for col in row.index if col.startswith("Prediction_d_")]
            if not daily_preds:
                return

            max_pred = max(daily_preds)
            min_pred = min(daily_preds)

            # Calcular condiciones para orden LONG.
            ideal_profit_pips_buy = (max_pred - current_price) / self.params.pip_cost
            ideal_drawdown_pips_buy = max((current_price - min_pred) / self.params.pip_cost, self.params.min_drawdown_pips)
            rr_buy = ideal_profit_pips_buy / ideal_drawdown_pips_buy if ideal_drawdown_pips_buy > 0 else 0
            tp_buy = current_price + self.params.tp_multiplier * ideal_profit_pips_buy * self.params.pip_cost
            sl_buy = current_price - self.params.sl_multiplier * ideal_drawdown_pips_buy * self.params.pip_cost

            # Calcular condiciones para orden SHORT.
            ideal_profit_pips_sell = (current_price - min_pred) / self.params.pip_cost
            ideal_drawdown_pips_sell = max((max_pred - current_price) / self.params.pip_cost, self.params.min_drawdown_pips)
            rr_sell = ideal_profit_pips_sell / ideal_drawdown_pips_sell if ideal_drawdown_pips_sell > 0 else 0
            tp_sell = current_price - self.params.tp_multiplier * ideal_profit_pips_sell * self.params.pip_cost
            sl_sell = current_price + self.params.sl_multiplier * ideal_drawdown_pips_sell * self.params.pip_cost

            # Imprimir métricas de entrada para debug.
            print(f"[DEBUG ENTRY] {dt} | Current Price: {current_price:.5f}", flush=True)
            print(f"[DEBUG ENTRY - LONG] Profit: {ideal_profit_pips_buy:.2f} pips, Risk: {ideal_drawdown_pips_buy:.2f} pips, RR: {rr_buy:.2f}, TP: {tp_buy:.5f}, SL: {sl_buy:.5f}", flush=True)
            print(f"[DEBUG ENTRY - SHORT] Profit: {ideal_profit_pips_sell:.2f} pips, Risk: {ideal_drawdown_pips_sell:.2f} pips, RR: {rr_sell:.2f}, TP: {tp_sell:.5f}, SL: {sl_sell:.5f}", flush=True)

            # Determinar la señal (entrada long o short) según cuál tenga mayor RR y supere el profit_threshold.
            if (ideal_profit_pips_buy >= self.params.profit_threshold) and (rr_buy >= rr_sell):
                signal = 'long'
                self.current_tp = tp_buy
                self.current_sl = sl_buy
            elif (ideal_profit_pips_sell >= self.params.profit_threshold) and (rr_sell > rr_buy):
                signal = 'short'
                self.current_tp = tp_sell
                self.current_sl = sl_sell
            else:
                return  # No se cumple la condición mínima de ganancia.

            # Almacenar métricas de entrada para debug en salida de trade.
            self.entry_profit = ideal_profit_pips_buy if signal == 'long' else ideal_profit_pips_sell
            self.entry_risk = ideal_drawdown_pips_buy if signal == 'long' else ideal_drawdown_pips_sell
            self.entry_rr = rr_buy if signal == 'long' else rr_sell
            self.entry_signal = signal

            # Calcular el tamaño de la orden.
            # Se usa una función de asignación proporcional según RR (usando min_order_volume y max_order_volume).
            order_size = self.compute_size(self.entry_rr)
            if order_size <= 0:
                print("[DEBUG] Order size <= 0, skipping trade", flush=True)
                return

            # Ejecutar la orden.
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

        def on_trade(self, trade):
            # Se llama cuando se cierra un trade.
            if trade.isclosed:
                duration = len(self) - (self.trade_entry_bar if hasattr(self, "trade_entry_bar") else 0)
                dt = self.data.index[-1]
                entry_price = self.order_entry_price if hasattr(self, "order_entry_price") and self.order_entry_price is not None else 0
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
                # Reset variables for the next trade.
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
