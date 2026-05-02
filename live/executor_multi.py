"""Live executor — runs the 4-model strategy on TopStepX funded account.

Funded account rules:
- $2,000 trailing drawdown (locks once balance reaches $52K)
- 90/10 profit split
- 5 winning days + 30 trading days for first $10K payout

Risk management:
- Daily R tracking with profit cap (1.8R) and loss cap (0.25R)
- Consecutive loss cooldown (2 losses → skip next)
- Per-model max daily trades
- Breakeven, partial exit, trailing stop management
- Time stop, session close flatten
- DD protection buffer ($1,700 of $2,000)
"""
from __future__ import annotations
import logging
import time
from datetime import datetime, timedelta, time as dt_time
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field
import pandas as pd
from config import Config
from data.loader import build_daily_bars
from strategy.multi import MultiModelGenerator
from strategy.models.base import Signal
from live.broker_topstep import TopStepBroker

log = logging.getLogger(__name__)

TICK_SIZE = 0.25
MNQ_TICK_VALUE = 0.50
CT = ZoneInfo('America/Chicago')


@dataclass
class LiveTrade:
    signal: Signal
    direction: str
    entry_price: float
    stop_price: float
    target_price: float
    risk: float
    entry_time: datetime
    contracts: int
    order_ids: dict = field(default_factory=dict)
    moved_be: bool = False
    partial_taken: bool = False
    trailing: bool = False
    mfe: float = 0.0


class LiveExecutor:
    def __init__(self, cfg: Config, broker: TopStepBroker, contracts: int = 20):
        self.cfg = cfg
        self.broker = broker
        self.contracts = contracts

        self.buf = pd.DataFrame()
        self.daily_df = pd.DataFrame()
        self.gen = MultiModelGenerator(cfg)
        self.last_signal_key = None
        self.trade: LiveTrade | None = None

        self.daily_r = 0.0
        self.daily_pnl_usd = 0.0
        self.consec_losses = 0
        self.daily_model_count = {}
        self.cur_date = None
        self.peak_balance = None
        self.start_balance = None
        self.winning_days = 0
        self.total_days = 0

        self.profit_cap_r = 1.8
        self.loss_cap_r = 0.25
        self.dd_protect_usd = 1700

    def run(self):
        log.info("Loading historical bars for warmup...")
        self.buf = self.broker.get_bars(minutes_back=5000)
        log.info(f"Loaded {len(self.buf)} bars "
                 f"({self.buf['datetime'].min()} → {self.buf['datetime'].max()})")

        self.daily_df = build_daily_bars(self.buf)
        self.daily_df['date'] = pd.to_datetime(self.daily_df['date']).dt.date

        acct = self.broker.get_account_info()
        self.start_balance = acct.get('balance', 50000)
        self.peak_balance = self.start_balance
        log.info(f"Account balance: ${self.start_balance:,.0f}")

        log.info(f"Strategy active — {self.contracts} MNQ | "
                 f"DD protect ${self.dd_protect_usd} | "
                 f"Profit cap {self.profit_cap_r}R | Loss cap {self.loss_cap_r}R")
        log.info("Waiting for signals...\n")

        while True:
            try:
                self._tick()
            except KeyboardInterrupt:
                raise
            except Exception:
                log.exception("Tick error")
            time.sleep(30)

    def _tick(self):
        now = datetime.now(CT)
        today = now.date()

        if today != self.cur_date:
            self._new_day(today)

        if now.time() < dt_time(8, 30):
            return
        if now.time() >= dt_time(14, 55):
            if self.trade:
                log.info("Session close — flattening")
                self._close_trade('session_close')
            return

        latest = self.broker.get_latest_bars(10)
        if latest.empty:
            return
        new_bars = self._merge_bars(latest)
        if new_bars == 0:
            return

        self.daily_df = build_daily_bars(self.buf)
        self.daily_df['date'] = pd.to_datetime(self.daily_df['date']).dt.date

        if self.trade:
            self._manage_trade()
        else:
            self._check_signals()

    def _new_day(self, today):
        if self.cur_date is not None:
            self.total_days += 1
            if self.daily_pnl_usd > 0:
                self.winning_days += 1

        self.cur_date = today
        self.daily_r = 0.0
        self.daily_pnl_usd = 0.0
        self.daily_model_count = {}

        log.info(f"\n{'='*55}")
        log.info(f"New day: {today}")

        acct = self.broker.get_account_info()
        bal = acct.get('balance', self.start_balance)
        if bal > self.peak_balance:
            self.peak_balance = bal
        dd = self.peak_balance - bal
        total_pnl = bal - self.start_balance

        log.info(f"Balance: ${bal:,.0f} | P&L: ${total_pnl:+,.0f} | "
                 f"Peak: ${self.peak_balance:,.0f}")
        log.info(f"DD: ${dd:,.0f} / ${self.dd_protect_usd} | "
                 f"Win days: {self.winning_days} | "
                 f"Trading days: {self.total_days}")
        log.info(f"{'='*55}\n")

    def _merge_bars(self, latest: pd.DataFrame) -> int:
        if self.buf.empty:
            self.buf = latest
            return len(latest)
        last_time = self.buf['datetime'].max()
        new = latest[latest['datetime'] > last_time]
        if new.empty:
            return 0
        self.buf = pd.concat([self.buf, new], ignore_index=True)
        if len(self.buf) > 3000:
            self.buf = self.buf.iloc[-3000:].reset_index(drop=True)
        return len(new)

    def _check_signals(self):
        if self.daily_r >= self.profit_cap_r:
            return
        if self.daily_r <= -self.loss_cap_r:
            return
        if self.consec_losses >= self.cfg.risk.consec_loss_cooldown:
            self.consec_losses = 0
            log.info("Cooldown triggered — skipping signal")
            return

        acct = self.broker.get_account_info()
        bal = acct.get('balance', self.start_balance)
        if bal > self.peak_balance:
            self.peak_balance = bal
        dd = self.peak_balance - bal
        if dd >= self.dd_protect_usd:
            log.warning(f"DD protection — ${dd:,.0f} drawdown, limit ${self.dd_protect_usd}")
            return

        try:
            signals = self.gen.generate(self.buf, self.daily_df, None)
        except Exception:
            log.exception("Signal generation failed")
            return

        if not signals:
            return

        sig = signals[-1]
        sig_key = f"{sig.model}_{sig.ts}_{sig.direction}"
        if sig_key == self.last_signal_key:
            return

        now = datetime.now(CT)
        sig_age = (now - sig.ts.to_pydatetime().replace(tzinfo=CT)).total_seconds()
        if sig_age > 120:
            return

        model_key = (self.cur_date, sig.model)
        rp = sig.risk_profile
        if rp and self.daily_model_count.get(model_key, 0) >= rp.max_daily:
            return

        self.last_signal_key = sig_key
        self._enter_trade(sig)

    def _enter_trade(self, sig: Signal):
        risk = abs(sig.entry - sig.stop)
        if risk <= 0:
            return

        log.info(f"\n>>> SIGNAL: [{sig.model}] {sig.direction.upper()} "
                 f"@ {sig.entry:.2f}")
        log.info(f"    Stop: {sig.stop:.2f} | Target: {sig.target:.2f} | "
                 f"RR: {sig.rr:.1f} | Risk: {sig.risk_ticks:.0f} ticks")

        try:
            ids = self.broker.place_bracket(
                direction=sig.direction,
                qty=self.contracts,
                stop_price=sig.stop,
                target_price=sig.target,
            )
        except Exception:
            log.exception("Order placement failed")
            return

        rp = sig.risk_profile
        time_stop = rp.time_stop_minutes if rp else self.cfg.strategy.time_stop_minutes

        self.trade = LiveTrade(
            signal=sig,
            direction=sig.direction,
            entry_price=sig.entry,
            stop_price=sig.stop,
            target_price=sig.target,
            risk=risk,
            entry_time=datetime.now(CT),
            contracts=self.contracts,
            order_ids=ids,
        )

        model_key = (self.cur_date, sig.model)
        self.daily_model_count[model_key] = \
            self.daily_model_count.get(model_key, 0) + 1

        log.info(f"    ENTERED — {self.contracts} MNQ | "
                 f"Time stop: {time_stop} min")

    def _manage_trade(self):
        t = self.trade
        if not t:
            return

        pos = self.broker.position_size()
        if pos == 0:
            self._on_trade_closed()
            return

        bar = self.buf.iloc[-1]
        is_long = t.direction == 'long'
        rp = t.signal.risk_profile

        be_trigger = rp.be_trigger_rr if rp else self.cfg.risk.be_trigger_rr
        partial_rr = rp.partial_rr if rp else self.cfg.risk.partial_rr
        trail_pct = rp.trail_pct if rp else 0.0
        time_stop_min = rp.time_stop_minutes if rp else self.cfg.strategy.time_stop_minutes

        if is_long:
            best = bar['high'] - t.entry_price
        else:
            best = t.entry_price - bar['low']

        if best > t.mfe:
            t.mfe = best
            if t.trailing and trail_pct > 0:
                trail_dist = trail_pct * t.risk
                if is_long:
                    new_stop = t.entry_price + t.mfe - trail_dist
                    if new_stop > t.stop_price:
                        t.stop_price = round(new_stop / TICK_SIZE) * TICK_SIZE
                        self.broker.modify_stop(t.stop_price)
                else:
                    new_stop = t.entry_price - t.mfe + trail_dist
                    if new_stop < t.stop_price:
                        t.stop_price = round(new_stop / TICK_SIZE) * TICK_SIZE
                        self.broker.modify_stop(t.stop_price)

        if not t.partial_taken and t.risk > 0 and best >= t.risk * partial_rr:
            t.partial_taken = True
            if trail_pct > 0:
                t.trailing = True
            log.info(f"    Partial trigger hit ({partial_rr}R)")

        if not t.moved_be and t.risk > 0 and best >= t.risk * be_trigger:
            t.moved_be = True
            t.stop_price = t.entry_price
            self.broker.modify_stop(t.entry_price)
            log.info(f"    Moved stop to BREAKEVEN @ {t.entry_price:.2f}")

        elapsed = (datetime.now(CT) - t.entry_time).total_seconds() / 60
        if elapsed >= time_stop_min and not t.moved_be:
            log.info(f"    TIME STOP ({time_stop_min} min)")
            self._close_trade('time_stop')
            return

        if datetime.now(CT).time() >= dt_time(14, 55):
            log.info("    SESSION CLOSE")
            self._close_trade('session_close')

    def _on_trade_closed(self):
        t = self.trade
        if not t:
            return

        self.broker.cancel_all_exit_orders()

        try:
            trades = self.broker._post('/api/Trade/search', {
                'accountId': self.broker.account_id,
                'startTimestamp': t.entry_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            })
            trade_list = trades.get('trades', [])
            pnl = sum(tr.get('profitAndLoss', 0) for tr in trade_list
                      if tr.get('contractId') == self.broker.contract_id)
        except Exception:
            pnl = 0

        risk_usd = t.risk * t.contracts * MNQ_TICK_VALUE / TICK_SIZE
        total_r = pnl / risk_usd if risk_usd > 0 else 0

        self.daily_r += total_r
        self.daily_pnl_usd += pnl
        if total_r <= -0.5:
            self.consec_losses += 1
        else:
            self.consec_losses = 0

        reason = 'target' if total_r > 0.5 else ('stop' if total_r < -0.5 else 'breakeven')
        log.info(f"\n    CLOSED — {reason} | {total_r:+.2f}R (${pnl:+,.0f})")
        log.info(f"    Daily: {self.daily_r:+.2f}R (${self.daily_pnl_usd:+,.0f}) | "
                 f"Consec losses: {self.consec_losses}")

        self.broker._stop_order_id = None
        self.broker._target_order_id = None
        self.broker._entry_order_id = None
        self.trade = None

    def _close_trade(self, reason: str):
        log.info(f"    Closing trade: {reason}")
        self.broker.flatten()

        t = self.trade
        if t:
            bar = self.buf.iloc[-1] if not self.buf.empty else None
            if bar is not None:
                is_long = t.direction == 'long'
                if is_long:
                    raw_pnl = (bar['close'] - t.entry_price)
                else:
                    raw_pnl = (t.entry_price - bar['close'])
                total_r = raw_pnl / t.risk if t.risk > 0 else 0
            else:
                total_r = 0

            self.daily_r += total_r
            pnl_usd = total_r * t.risk * t.contracts * MNQ_TICK_VALUE / TICK_SIZE
            self.daily_pnl_usd += pnl_usd
            if total_r <= -0.5:
                self.consec_losses += 1
            else:
                self.consec_losses = 0

            log.info(f"    Result: {total_r:+.2f}R (${pnl_usd:+,.0f}) | "
                     f"Daily: {self.daily_r:+.2f}R (${self.daily_pnl_usd:+,.0f})")

        self.broker._stop_order_id = None
        self.broker._target_order_id = None
        self.broker._entry_order_id = None
        self.trade = None

    def shutdown(self):
        if self.trade:
            log.info("Shutdown — flattening open position")
            self.broker.flatten()
        log.info("Executor shutdown complete.")
