"""Multi-model signal orchestrator — generates, filters, and resolves signals."""
from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import time as dt_time
from config import Config
from strategy.vwap import compute_vwap, compute_opening_range
from strategy.models.base import Signal
from strategy.models import ALL_MODELS


class MultiModelGenerator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tick = cfg.instrument.tick_size
        self.models = [ModelClass(cfg) for ModelClass in ALL_MODELS]

    def generate(self, df: pd.DataFrame, daily: pd.DataFrame,
                 df_es: pd.DataFrame | None = None) -> list[Signal]:
        df = compute_vwap(df)
        df = compute_opening_range(df, minutes=15)
        df = df.reset_index(drop=True)

        from strategy.quant import compute_all_quant_features
        df = compute_all_quant_features(df)

        context = self._build_context(daily)

        all_signals: list[Signal] = []
        for model in self.models:
            sigs = model.generate(df, daily, context)
            all_signals.extend(sigs)

        filtered = [s for s in all_signals if s.ts.time() < dt_time(14, 30)]
        filtered.sort(key=lambda s: s.idx)
        resolved = self._resolve_conflicts(filtered)
        return resolved

    def _build_context(self, daily: pd.DataFrame) -> dict:
        daily_map = {}
        regime_map = {}
        daily_s = daily.sort_values('date').reset_index(drop=True)
        closes = daily_s['close'].values
        ema20 = self._ema(closes, 20)
        ema50 = self._ema(closes, 50)

        for i in range(1, len(daily_s)):
            d = pd.Timestamp(daily_s.iloc[i]['date']).date()
            prev = daily_s.iloc[i - 1]
            daily_map[d] = {
                'pdh': prev['high'], 'pdl': prev['low'],
                'prev_close': prev['close'],
            }
            if i >= 50:
                c = prev['close']
                above20 = c > ema20[i - 1]
                above50 = c > ema50[i - 1]
                if above20 and above50:
                    regime_map[d] = 'bull'
                elif not above20 and not above50:
                    regime_map[d] = 'bear'
                else:
                    regime_map[d] = 'chop'

        return {'daily_map': daily_map, 'regime_map': regime_map}

    @staticmethod
    def _ema(data, span):
        out = np.empty_like(data, dtype=float)
        out[0] = data[0]
        k = 2.0 / (span + 1)
        for i in range(1, len(data)):
            out[i] = data[i] * k + out[i - 1] * (1 - k)
        return out

    @staticmethod
    def _resolve_conflicts(signals: list[Signal], cooldown_bars: int = 5) -> list[Signal]:
        if not signals:
            return []

        resolved = [signals[0]]
        for sig in signals[1:]:
            last = resolved[-1]
            if sig.idx - last.idx < cooldown_bars:
                if sig.priority < last.priority:
                    resolved[-1] = sig
                elif sig.priority == last.priority and sig.rr > last.rr:
                    resolved[-1] = sig
            else:
                resolved.append(sig)
        return resolved
