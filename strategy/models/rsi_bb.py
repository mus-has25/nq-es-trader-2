"""RSI + Bollinger Bands Band-Rejection Model (5M)

Entry conditions:
  LONG : RSI(14) < 40  (oversold)
         AND bar low pierces lower BB(20, 2σ) but close snaps back ABOVE it
         AND RSI is rising on signal bar (momentum turning)
         AND previous bar was bearish (selling exhaustion)
         AND bull regime (daily EMA20 > EMA50)

  SHORT: RSI(14) > 60  (overbought)
         AND bar high pierces upper BB but close snaps back BELOW it
         AND RSI is falling on signal bar
         AND previous bar was bullish
         AND bear regime

Session: Full RTH 9:30–16:00 ET on 5M bars
Optimised on 2025-2026 NQ data: 35 trades, 60.0% WR, 2.53 PF
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from strategy.models.base import BaseModel, ModelRiskProfile, Signal
from config import Config

# ── Tunable parameters ─────────────────────────────────────────────────────────
RSI_PERIOD    = 14
RSI_OS        = 40     # oversold: RSI must be below this for longs
RSI_OB        = 60     # overbought: RSI must be above this for shorts
BB_PERIOD     = 20
BB_STD        = 2.0
HAMMER_PCT    = 0.60   # close must be in top 60% of bar range (longs)
MIN_RR        = 1.5
MIN_RISK_TICKS = 8
MAX_RISK_TICKS = 80
TARGET_RR     = 2.0    # nominal target, may be capped at BB midline
MAX_DAILY     = 3
# ───────────────────────────────────────────────────────────────────────────────

_SESSION_S = 9 * 60 + 30    # 9:30 ET (RTH open)
_SESSION_E = 16 * 60 + 0    # 4:00 PM ET


class RSIBollingerModel(BaseModel):
    name = "rsi_bb"
    priority = 30

    def __init__(self, cfg: Config):
        rp = ModelRiskProfile(
            min_risk_ticks=MIN_RISK_TICKS, max_risk_ticks=MAX_RISK_TICKS,
            min_rr=MIN_RR,
            be_trigger_rr=1.0, partial_rr=1.0, partial_pct=0.5,
            time_stop_minutes=30, max_daily=MAX_DAILY,
            trail_pct=0.0,
        )
        super().__init__(cfg, rp)

    # ── Main ────────────────────────────────────────────────────────────────────
    def generate(self, df: pd.DataFrame, daily: pd.DataFrame,
                 context: dict) -> list[Signal]:
        # Auto-resample 1M → 5M when live data is passed at 1-minute resolution
        median_gap = df['datetime'].diff().dropna().median()
        if pd.notna(median_gap) and median_gap.total_seconds() < 290:
            df_orig = df.reset_index(drop=True)
            df = (df.set_index('datetime')[['open','high','low','close','volume']]
                  .resample('5min')
                  .agg({'open':'first','high':'max','low':'min',
                        'close':'last','volume':'sum'})
                  .dropna(subset=['open'])
                  .reset_index())
            df['symbol'] = df_orig['symbol'].iloc[0]
            need_remap = True
        else:
            df_orig    = None
            need_remap = False

        signals = self._generate_5m(df, context)

        if need_remap and signals:
            signals = self._remap_to_1m(signals, df_orig)

        return signals

    def _remap_to_1m(self, signals: list[Signal],
                     df1m: pd.DataFrame) -> list[Signal]:
        remapped = []
        for sig in signals:
            mask = df1m['datetime'] >= sig.ts
            if not mask.any():
                continue
            new_idx = int(df1m.index[mask][0])
            remapped.append(Signal(
                idx=new_idx, ts=sig.ts, model=sig.model,
                direction=sig.direction, entry=sig.entry,
                stop=sig.stop, target=sig.target,
                risk_ticks=sig.risk_ticks, reward_ticks=sig.reward_ticks,
                rr=sig.rr, tag=sig.tag, priority=sig.priority,
                risk_profile=sig.risk_profile,
            ))
        return remapped

    def _generate_5m(self, df: pd.DataFrame, context: dict) -> list[Signal]:
        daily_map  = context.get('daily_map', {})
        regime_map = context.get('regime_map', {})

        n    = len(df)
        tick = self.tick

        closes = df['close'].values.astype(float)
        highs  = df['high'].values.astype(float)
        lows   = df['low'].values.astype(float)
        opens  = df['open'].values.astype(float)

        # ── 1. RSI(14) — Wilder smoothing ────────────────────────────────────
        rsi = self._rsi(closes, RSI_PERIOD)

        # ── 2. Bollinger Bands(20, 2σ) ────────────────────────────────────────
        s = pd.Series(closes)
        bb_mid   = s.rolling(BB_PERIOD).mean().values
        bb_std   = s.rolling(BB_PERIOD).std(ddof=0).values
        bb_upper = bb_mid + BB_STD * bb_std
        bb_lower = bb_mid - BB_STD * bb_std

        # ── 3. Session mask ───────────────────────────────────────────────────
        t_mins   = (df['datetime'].dt.hour.values * 60 +
                    df['datetime'].dt.minute.values).astype(int)
        date_arr = df['datetime'].dt.date.values

        in_session = (t_mins >= _SESSION_S) & (t_mins < _SESSION_E)
        min_start  = max(RSI_PERIOD + 2, BB_PERIOD + 1)
        session_idx = np.where(in_session)[0]
        session_idx = session_idx[session_idx >= min_start]

        signals: list[Signal] = []
        daily_used: dict = {}

        for idx in session_idx:
            d = date_arr[idx]
            if d not in daily_map:
                continue
            if daily_used.get(d, 0) >= MAX_DAILY:
                continue

            regime = regime_map.get(d, 'chop')
            r   = rsi[idx]
            cl  = closes[idx]
            hi  = highs[idx]
            lo  = lows[idx]
            bbl = bb_lower[idx]
            bbu = bb_upper[idx]
            bbm = bb_mid[idx]

            if np.isnan(r) or np.isnan(bbl):
                continue

            bar_range = hi - lo
            if bar_range < 1e-9:
                continue
            close_pct = (cl - lo) / bar_range

            # ── LONG: RSI oversold, BB lower band rejection ───────────────────
            if (r < RSI_OS
                    and lo < bbl               # wick pierces lower band
                    and cl > bbl               # close snaps back above
                    and close_pct >= HAMMER_PCT
                    and regime == 'bull'        # bull regime only
                    and rsi[idx] > rsi[idx-1]  # RSI momentum turning up
                    and opens[idx-1] > closes[idx-1]):  # prev bar bearish

                entry  = cl
                stop   = lo - 2 * tick
                risk   = entry - stop
                if risk <= 0:
                    continue
                risk_ticks = risk / tick
                if risk_ticks < MIN_RISK_TICKS or risk_ticks > MAX_RISK_TICKS:
                    continue
                target = max(entry + risk * TARGET_RR, bbm)
                reward = target - entry
                rr     = reward / risk
                if rr < MIN_RR:
                    continue

                signals.append(self._make_signal(
                    idx, df.iloc[idx], 'long', entry, stop, target, 'rsi_bb_long'))
                daily_used[d] = daily_used.get(d, 0) + 1

            # ── SHORT: RSI overbought, BB upper band rejection ────────────────
            elif (r > RSI_OB
                    and hi > bbu               # wick pierces upper band
                    and cl < bbu               # close snaps back below
                    and close_pct <= (1 - HAMMER_PCT)
                    and regime == 'bear'        # bear regime only
                    and rsi[idx] < rsi[idx-1]  # RSI momentum turning down
                    and opens[idx-1] < closes[idx-1]):  # prev bar bullish

                entry  = cl
                stop   = hi + 2 * tick
                risk   = stop - entry
                if risk <= 0:
                    continue
                risk_ticks = risk / tick
                if risk_ticks < MIN_RISK_TICKS or risk_ticks > MAX_RISK_TICKS:
                    continue
                target = min(entry - risk * TARGET_RR, bbm)
                reward = entry - target
                rr     = reward / risk
                if rr < MIN_RR:
                    continue

                signals.append(self._make_signal(
                    idx, df.iloc[idx], 'short', entry, stop, target, 'rsi_bb_short'))
                daily_used[d] = daily_used.get(d, 0) + 1

        return signals

    # ── RSI helper (Wilder smoothing) ──────────────────────────────────────────
    @staticmethod
    def _rsi(closes: np.ndarray, period: int) -> np.ndarray:
        n   = len(closes)
        out = np.full(n, np.nan)
        if n < period + 2:
            return out
        delta  = np.diff(closes)
        gains  = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)
        avg_g  = gains[:period].mean()
        avg_l  = losses[:period].mean()
        for i in range(period, n - 1):
            avg_g = (avg_g * (period - 1) + gains[i]) / period
            avg_l = (avg_l * (period - 1) + losses[i]) / period
            rs    = avg_g / avg_l if avg_l > 0 else 1e9
            out[i + 1] = 100 - (100 / (1 + rs))
        return out
