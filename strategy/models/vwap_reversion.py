"""VWAP Mean Reversion — fade extended moves beyond VWAP bands back toward VWAP."""
from __future__ import annotations
import pandas as pd
from datetime import time as dt_time
from strategy.models.base import BaseModel, ModelRiskProfile, Signal
from config import Config


class VWAPReversionModel(BaseModel):
    name = 'vwap_rev'
    priority = 20

    def __init__(self, cfg: Config):
        rp = ModelRiskProfile(
            min_risk_ticks=20, max_risk_ticks=100, min_rr=1.3,
            be_trigger_rr=1.0, partial_rr=1.0, partial_pct=0.3,
            time_stop_minutes=30, max_daily=5,
            trail_pct=0.0,
        )
        super().__init__(cfg, rp)

    def generate(self, df: pd.DataFrame, daily: pd.DataFrame,
                 context: dict) -> list[Signal]:
        regime_map = context['regime_map']
        daily_map = context['daily_map']

        signals: list[Signal] = []
        cur_date = None
        used = 0

        for idx in range(30, len(df)):
            bar = df.iloc[idx]
            dt = bar['datetime']
            d = dt.date()
            t = dt.time()

            if d != cur_date:
                cur_date = d
                used = 0

            if d not in daily_map or d not in regime_map:
                continue
            if used >= self.risk_profile.max_daily:
                continue
            if not (dt_time(9, 50) <= t < dt_time(15, 0)):
                continue
            if dt_time(11, 30) <= t < dt_time(13, 0):
                continue

            vwap = bar.get('vwap')
            vwap_std = bar.get('vwap_std')
            if pd.isna(vwap) or pd.isna(vwap_std) or vwap_std < 2 * self.tick:
                continue

            regime = regime_map[d]
            lower_band = vwap - 1.5 * vwap_std

            prev1 = df.iloc[idx - 1]

            # Long: price extended below VWAP, reversal candle
            if (prev1['low'] <= lower_band
                    and bar['close'] > bar['open']
                    and bar['close'] > lower_band):

                if regime == 'bear':
                    continue

                entry = bar['close']
                extreme_low = min(bar['low'], prev1['low'])
                stop = extreme_low - 2 * self.tick
                risk = entry - stop
                target = vwap
                reward = target - entry

                if reward > 0 and self._risk_ok(risk, reward):
                    signals.append(self._make_signal(
                        idx, bar, 'long', entry, stop, target,
                        'vwap_rev_long'))
                    used += 1

        return signals
