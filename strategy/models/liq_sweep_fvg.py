"""ICT Liquidity Sweep + FVG  (v3)

Long-only — sweep sell-side liquidity (swing low), enter on Fair Value Gap
retest during the NY AM kill-zone.

Design:
  • Sweeps detected at ANY time (including overnight / London)
  • FVG searched in bars immediately AFTER the sweep bar (+1 to +7)
  • Entry retest restricted to RTH AM session (9:45–11:45 ET)
  • Regime-aware: skip bear days
  • FVG-filled guard: skip if price already closed below the FVG zone
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import time as dt_time
from strategy.models.base import BaseModel, ModelRiskProfile, Signal
from config import Config

# ── Tunable parameters ─────────────────────────────────────────────────────────
PIVOT_BARS     = 15    # bars each side for swing low (bigger = real liquidity)
MAX_PIVOT_AGE  = 80    # max bars between pivot and sweep
FVG_POST       = 7     # bars AFTER sweep to search for FVG
MAX_SWEEP_AGE  = 55    # bars from sweep to AM entry
MIN_BARS_AFTER = 3     # entry at least N bars after sweep
MIN_FVG_TICKS  = 3     # minimum FVG width in ticks
MIN_SWEEP_TICKS= 4     # sweep must clear pivot by at least N ticks
DISPLACE_ATR_X = 1.1   # sweep bar range >= N × ATR(10)
SWEEP_CLOSE_PCT= 0.50  # sweep bar close in top X% of range (hammer quality)
TARGET_RR      = 2.0   # risk multiple for target
MIN_RR         = 1.5
MIN_RISK_TICKS = 8
MAX_RISK_TICKS = 80
MAX_DAILY      = 3

# Entry window: NY AM kill-zone (9:45 – 11:45 ET)
_ENTRY_S = 9 * 60 + 45
_ENTRY_E = 11 * 60 + 45
_SWEEP_MAX_T = 99 * 60   # no sweep timing restriction
# ───────────────────────────────────────────────────────────────────────────────


class LiquiditySweepFVGModel(BaseModel):
    name = "liq_sweep"
    priority = 10

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
        daily_map  = context.get('daily_map', {})
        regime_map = context.get('regime_map', {})

        n    = len(df)
        tick = self.tick

        highs  = df['high'].values.astype(float)
        lows   = df['low'].values.astype(float)
        closes = df['close'].values.astype(float)
        opens  = df['open'].values.astype(float)

        t_mins   = (df['datetime'].dt.hour.values * 60 +
                    df['datetime'].dt.minute.values).astype(int)
        date_arr = df['datetime'].dt.date.values

        # ── 1. Pivot lows (vectorised rolling min) ────────────────────────────
        win = 2 * PIVOT_BARS + 1
        roll_min = pd.Series(lows).rolling(win, center=True).min().values
        pivot_low_mask = (np.abs(lows - roll_min) < 1e-9) & ~np.isnan(roll_min)

        pl_val_s = pd.Series(np.where(pivot_low_mask, lows, np.nan))
        pl_idx_s = pd.Series(np.where(pivot_low_mask, np.arange(n), np.nan))
        last_pl_val = pl_val_s.ffill().shift(1).values
        last_pl_idx = (pl_idx_s.ffill().shift(1).fillna(-999).astype(int).values)

        # ── 2. ATR(10) ─────────────────────────────────────────────────────────
        atr10 = pd.Series(np.maximum(highs - lows, 1e-6)).rolling(10).mean().values

        # ── 3. Vectorised sweep detection (all hours) ─────────────────────────
        bar_idx  = np.arange(n)
        valid_pl = (
            (~np.isnan(last_pl_val)) &
            (last_pl_idx >= 0) &
            (bar_idx - last_pl_idx <= MAX_PIVOT_AGE)
        )
        bar_range = highs - lows
        bar_range_safe = np.where(bar_range > 1e-9, bar_range, 1.0)
        close_pct = np.where(bar_range > 1e-9, (closes - lows) / bar_range_safe, 0.5)
        ssl_mask = (
            valid_pl &
            (lows < last_pl_val - MIN_SWEEP_TICKS * tick) &
            (closes > last_pl_val) &
            (closes > opens) &
            (bar_range >= DISPLACE_ATR_X * atr10) &
            (close_pct >= SWEEP_CLOSE_PCT)
        )
        ssl_indices = np.where(ssl_mask)[0]

        # ── 4. Find FVG AFTER each sweep; also enforce sweep timing ───────────
        ssl_with_fvg: list[tuple[int, float, float, float]] = []
        for si in ssl_indices:
            # Sweep must happen before NY AM session starts
            if t_mins[si] > _SWEEP_MAX_T:
                continue
            sl   = lows[si]
            fstart = si + 1
            fend   = min(si + FVG_POST, n - 3)
            fvg    = self._find_bullish_fvg(highs, lows, fstart, fend, tick)
            if fvg is not None:
                ssl_with_fvg.append((si, sl, fvg[0], fvg[1]))

        if not ssl_with_fvg:
            return []

        # ── 5. Entry scan — AM kill-zone only ─────────────────────────────────
        in_session = (t_mins >= _ENTRY_S) & (t_mins < _ENTRY_E)
        min_start   = PIVOT_BARS + FVG_POST + 10
        session_idx = np.where(in_session)[0]
        session_idx = session_idx[session_idx >= min_start]

        signals: list[Signal] = []
        used_sweeps: set[int] = set()
        daily_used: dict = {}

        ssl_ptr   = 0
        active_ssl: list[tuple[int, float, float, float]] = []

        for idx in session_idx:
            d = date_arr[idx]
            if d not in daily_map:
                continue

            # Advance pointer — add sweeps that happened before current bar
            while (ssl_ptr < len(ssl_with_fvg)
                   and ssl_with_fvg[ssl_ptr][0] < idx):
                entry = ssl_with_fvg[ssl_ptr]
                if entry[0] not in used_sweeps:
                    active_ssl.append(entry)
                ssl_ptr += 1

            # Expire stale
            if active_ssl:
                active_ssl = [(si, sl, fl, fh) for si, sl, fl, fh in active_ssl
                              if idx - si <= MAX_SWEEP_AGE
                              and si not in used_sweeps]
            if not active_ssl:
                continue
            if daily_used.get(d, 0) >= MAX_DAILY:
                continue

            regime = regime_map.get(d, 'chop')
            if regime != 'bull':
                continue

            lo = lows[idx]
            cl = closes[idx]
            op = opens[idx]

            if cl <= op:   # require bullish close on entry bar
                continue

            # minimum gap between sweep and entry (let price settle)
            # checked per setup below

            pdh = daily_map[d]['pdh']

            for si, sl, fvg_lo, fvg_hi in active_ssl:
                # Minimum separation between sweep and entry
                if idx - si < MIN_BARS_AFTER:
                    continue

                # Retest: bar dipped into FVG zone, closed above lower bound
                if lo <= fvg_hi and cl >= fvg_lo:
                    entry  = cl
                    stop   = sl - 2 * tick
                    risk   = entry - stop
                    if risk <= 0:
                        continue
                    target = entry + risk * TARGET_RR
                    if pdh > entry:
                        target = max(target, pdh)
                    reward = target - entry

                    if self._risk_ok(risk, reward):
                        signals.append(self._make_signal(
                            idx, df.iloc[idx], 'long', entry, stop, target,
                            'liq_sweep_long'))
                        used_sweeps.add(si)
                        daily_used[d] = daily_used.get(d, 0) + 1
                        break

        return signals

    # ── FVG helper ──────────────────────────────────────────────────────────────
    @staticmethod
    def _find_bullish_fvg(highs, lows, start, end, tick):
        """Return (zone_lo, zone_hi) of first bullish FVG in bar range [start, end-2].
        A bullish FVG: candle1.high < candle3.low (gap between them)."""
        best = None
        best_size = MIN_FVG_TICKS * tick - 1e-9
        for i in range(start, end - 1):
            zone_lo = highs[i]
            zone_hi = lows[i + 2]
            size    = zone_hi - zone_lo
            if size > best_size:
                best_size = size
                best = (zone_lo, zone_hi)
        return best
