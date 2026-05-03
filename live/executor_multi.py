"""Live executor — runs OU+Trend+VWAP strategy on TopStepX funded account.

Schedule: Mon-Fri mornings, model-specific sizing (OU:12 T:14 V:8 MNQ).

Adaptive withdrawals: extract $500-$2K whenever balance exceeds
DD floor + $2K buffer. Requires 5 winning days ($150+) per TopStepX.

Risk controls:
- Model-specific position sizing: OU:12, Trend:14, VWAP:8 MNQ
- Slippage size reduction: 50% size for trades under 50 ticks risk
- $600 prospective daily loss cap: skip trades if worst-case would breach
- Progressive DD scaling: reduce to 50% as DD grows from $1K to $1.5K
- Streak protection: 75% size after 2 consecutive losing days

Funded account rules (TopStepX Express Funded 50K):
- $2,000 EOD trailing drawdown (locks at $50K floor once peak hits $52K)
- $1,000 daily loss limit (auto-liquidation, not permanent)
- 90/10 profit split
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
from live.broker_topstep import TopStepBroker, ORD_FILLED, ORD_CANCELLED, ORD_REJECTED, ORD_EXPIRED

log = logging.getLogger(__name__)

TICK_SIZE = 0.25
MNQ_TICK_VALUE = 0.50
CT = ZoneInfo('America/Chicago')


ENTRY_TIMEOUT_SEC = 60


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
    pending: bool = True
    moved_be: bool = False
    partial_taken: bool = False
    trailing: bool = False
    mfe: float = 0.0


class LiveExecutor:
    def __init__(self, cfg: Config, broker: TopStepBroker,
                 model_qty: dict | None = None, phase: str = 'eval'):
        self.cfg = cfg
        self.broker = broker
        self.model_qty = model_qty or {'ou_rev': 12, 'trend': 14, 'vwap_rev': 8}
        self.phase = phase

        self.buf = pd.DataFrame()
        self.daily_df = pd.DataFrame()
        self.gen = MultiModelGenerator(cfg)
        self.last_signal_key = None
        self.trade: LiveTrade | None = None

        self.daily_r = 0.0
        self.daily_pnl_usd = 0.0
        self.daily_model_count = {}
        self.cur_date = None
        self.peak_balance = None
        self.start_balance = None
        self.winning_days = 0
        self.total_days = 0

        self.active_models = set(self.model_qty.keys())
        self.withdraw_buffer_usd = 2000
        self.min_withdraw_usd = 500
        self.slip_size_threshold = 50
        self.slip_size_pct = 0.50
        self.dd_floor = None
        self.dd_locked = False
        self.max_risk_ticks = 100
        self.max_trade_loss_usd = 500
        self.daily_loss_cap_usd = 600
        self.eval_target = 3000
        self.eval_day_profits = []
        self.consec_losing_days = 0
        self.streak_reduce_after = 2
        self.streak_reduce_pct = 0.75
        self.dd_scale_start = 1000
        self.dd_scale_floor = 0.50

    def run(self):
        log.info("Loading historical bars for warmup (need 50+ daily bars for regime)...")
        self.buf = self.broker.get_bars(minutes_back=120000)
        log.info(f"Loaded {len(self.buf)} bars "
                 f"({self.buf['datetime'].min()} → {self.buf['datetime'].max()})")

        self.daily_df = build_daily_bars(self.buf)
        self.daily_df['date'] = pd.to_datetime(self.daily_df['date']).dt.date
        n_daily = len(self.daily_df)
        log.info(f"Daily bars: {n_daily} (need 50+ for regime)")
        if n_daily < 50:
            log.warning(f"Only {n_daily} daily bars — regime map will be incomplete, "
                        f"some signals may not fire")

        acct = self.broker.get_account_info()
        self.start_balance = acct.get('balance', 50000)
        self.peak_balance = self.start_balance
        self.dd_floor = self.start_balance - 2000
        self.dd_locked = False
        log.info(f"Account balance: ${self.start_balance:,.0f}")
        log.info(f"Phase: {self.phase.upper()}")

        qty_str = ' | '.join(f"{m}:{q}" for m, q in sorted(self.model_qty.items()))
        log.info(f"Strategy active — {qty_str}")
        log.info(f"Risk: max trade loss ${self.max_trade_loss_usd} | "
                 f"daily cap -${self.daily_loss_cap_usd} | "
                 f"{self.slip_size_pct:.0%} size under {self.slip_size_threshold}t (slippage)")
        log.info(f"Streak: {self.streak_reduce_pct*100:.0f}% after "
                 f"{self.streak_reduce_after} losing days | "
                 f"DD scale: {self.dd_scale_floor*100:.0f}% @ ${self.dd_scale_start}+ DD")
        log.info(f"Withdraw: adaptive, ${self.min_withdraw_usd}+ when "
                 f"${self.withdraw_buffer_usd} buffer above DD floor")
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
                if self.phase == 'eval':
                    self.eval_day_profits.append(self.daily_pnl_usd)
            if self.daily_pnl_usd < 0:
                self.consec_losing_days += 1
            else:
                self.consec_losing_days = 0

        self.cur_date = today
        self.daily_r = 0.0
        self.daily_pnl_usd = 0.0
        self.daily_model_count = {}

        log.info(f"\n{'='*55}")
        log.info(f"New day: {today} [{self.phase.upper()}]")

        acct = self.broker.get_account_info()
        bal = acct.get('balance', self.start_balance)
        if bal > self.peak_balance:
            self.peak_balance = bal
        self._update_dd_floor()
        dd = self.peak_balance - bal
        cushion = bal - self.dd_floor
        total_pnl = bal - self.start_balance

        log.info(f"Balance: ${bal:,.0f} | P&L: ${total_pnl:+,.0f} | "
                 f"Peak: ${self.peak_balance:,.0f}")
        log.info(f"DD floor: ${self.dd_floor:,.0f} ({'locked' if self.dd_locked else 'trailing'}) | "
                 f"Cushion: ${cushion:,.0f} | "
                 f"Win days: {self.winning_days} | "
                 f"Trading days: {self.total_days}")

        if self.phase == 'eval' and self.eval_day_profits:
            total_positive = sum(self.eval_day_profits)
            max_day = max(self.eval_day_profits)
            consistency = max_day / total_positive if total_positive > 0 else 0
            remaining = max(0, self.eval_target - total_pnl)
            log.info(f"EVAL: ${total_pnl:+,.0f} / ${self.eval_target:,} target | "
                     f"Consistency: {consistency:.0%} (max day ${max_day:,.0f} / "
                     f"${total_positive:,.0f} total, need <50%)")
            if consistency > 0.40:
                log.warning(f"CONSISTENCY WARNING: {consistency:.0%} — "
                            f"approaching 50% limit")

        if self.consec_losing_days >= self.streak_reduce_after:
            log.info(f"STREAK ALERT: {self.consec_losing_days} consecutive losing days — "
                     f"reducing size to {self.streak_reduce_pct*100:.0f}%")
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
        if len(self.buf) > 30000:
            self.buf = self.buf.iloc[-30000:].reset_index(drop=True)
        return len(new)

    def _update_dd_floor(self):
        if self.phase == 'eval':
            self.dd_floor = self.peak_balance - 2000
        else:
            if not self.dd_locked and (self.peak_balance - self.start_balance) >= 2000:
                self.dd_locked = True
                self.dd_floor = self.peak_balance - 2000
            if not self.dd_locked:
                self.dd_floor = self.peak_balance - 2000

    def _check_signals(self):
        now = datetime.now(CT)
        if now.time() >= dt_time(14, 0):
            return

        try:
            signals = self.gen.generate(self.buf, self.daily_df, None)
        except Exception:
            log.exception("Signal generation failed")
            return

        if not signals:
            return

        signals = [s for s in signals if s.model in self.active_models]
        if not signals:
            return

        if now.weekday() == 2:
            signals = [s for s in signals if s.model != 'vwap_rev']
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

        max_qty = self.model_qty.get(sig.model, 0)
        if max_qty <= 0:
            return
        qty = min(max_qty,
                  int(self.max_trade_loss_usd / (sig.risk_ticks * MNQ_TICK_VALUE)))
        if qty < 1:
            qty = 1

        if sig.risk_ticks < self.slip_size_threshold:
            qty = max(1, int(qty * self.slip_size_pct))
            log.info(f"    SLIP SIZE — {sig.risk_ticks:.0f}t < {self.slip_size_threshold}t, "
                     f"{self.slip_size_pct:.0%} → {qty} MNQ")

        if self.consec_losing_days >= self.streak_reduce_after:
            qty = max(1, int(qty * self.streak_reduce_pct))
            log.info(f"    STREAK — {self.consec_losing_days} losing days, "
                     f"reduced to {qty} MNQ")

        acct = self.broker.get_account_info()
        bal = acct.get('balance', self.start_balance)
        dd = self.peak_balance - bal if self.peak_balance else 0

        if dd >= self.dd_scale_start:
            dd_ratio = min(1.0, (dd - self.dd_scale_start) / 500)
            scale = 1.0 - (dd_ratio * (1.0 - self.dd_scale_floor))
            qty = max(1, int(qty * scale))
            log.info(f"    DD SCALE — ${dd:,.0f} drawdown, "
                     f"scale {scale:.0%}, {qty} MNQ")

        potential_loss = sig.risk_ticks * qty * MNQ_TICK_VALUE
        if self.daily_pnl_usd - potential_loss < -self.daily_loss_cap_usd:
            log.info(f"    SKIP — daily cap: P&L ${self.daily_pnl_usd:,.0f}, "
                     f"potential loss ${potential_loss:,.0f} would breach "
                     f"-${self.daily_loss_cap_usd:,.0f} cap")
            return

        log.info(f"\n>>> SIGNAL: [{sig.model}] {sig.direction.upper()} "
                 f"@ {sig.entry:.2f} ({qty} MNQ)")
        log.info(f"    Stop: {sig.stop:.2f} | Target: {sig.target:.2f} | "
                 f"RR: {sig.rr:.1f} | Risk: {sig.risk_ticks:.0f} ticks | "
                 f"Max loss: ${potential_loss:,.0f}")

        try:
            entry_id = self.broker.place_limit_entry(
                direction=sig.direction,
                qty=qty,
                entry_price=sig.entry,
            )
        except Exception:
            log.exception("Order placement failed")
            return

        self.trade = LiveTrade(
            signal=sig,
            direction=sig.direction,
            entry_price=sig.entry,
            stop_price=sig.stop,
            target_price=sig.target,
            risk=risk,
            entry_time=datetime.now(CT),
            contracts=qty,
            order_ids={'entry': entry_id},
            pending=True,
        )

        model_key = (self.cur_date, sig.model)
        self.daily_model_count[model_key] = \
            self.daily_model_count.get(model_key, 0) + 1

        log.info(f"    PENDING — {qty} MNQ limit @ {sig.entry:.2f}")

    def _manage_trade(self):
        t = self.trade
        if not t:
            return

        if t.pending:
            self._check_entry_fill()
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

    def _check_entry_fill(self):
        t = self.trade
        entry_id = t.order_ids.get('entry')
        status = self.broker.get_order_status(entry_id)

        if status in (ORD_CANCELLED, ORD_REJECTED, ORD_EXPIRED):
            status_name = {ORD_CANCELLED: 'CANCELLED', ORD_REJECTED: 'REJECTED', ORD_EXPIRED: 'EXPIRED'}
            log.info(f"    ENTRY {status_name.get(status, 'REMOVED')} — resetting")
            self._reset_trade()
            return

        if status == ORD_FILLED:
            t.pending = False
            t.entry_time = datetime.now(CT)
            log.info(f"    FILLED — {t.contracts} MNQ @ {t.entry_price:.2f}")
            try:
                exit_ids = self.broker.place_exit_bracket(
                    direction=t.direction,
                    qty=t.contracts,
                    stop_price=t.stop_price,
                    target_price=t.target_price,
                )
                t.order_ids.update(exit_ids)
            except Exception:
                log.exception("Exit bracket placement failed — flattening")
                self.broker.flatten()
                self._reset_trade()
            return

        elapsed = (datetime.now(CT) - t.entry_time).total_seconds()
        if elapsed >= ENTRY_TIMEOUT_SEC:
            log.info(f"    ENTRY TIMEOUT — {elapsed:.0f}s, cancelling limit order")
            self.broker.cancel_order(entry_id)
            self._reset_trade()

    def _reset_trade(self):
        self.broker._stop_order_id = None
        self.broker._target_order_id = None
        self.broker._entry_order_id = None
        self.trade = None

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
            pnl = sum((tr.get('profitAndLoss') or 0) for tr in trade_list
                      if tr.get('contractId') == self.broker.contract_id)
        except Exception:
            pnl = 0

        risk_usd = t.risk * t.contracts * MNQ_TICK_VALUE / TICK_SIZE
        total_r = pnl / risk_usd if risk_usd > 0 else 0

        self.daily_r += total_r
        self.daily_pnl_usd += pnl

        reason = 'target' if total_r > 0.5 else ('stop' if total_r < -0.5 else 'breakeven')
        log.info(f"\n    CLOSED — {reason} | {total_r:+.2f}R (${pnl:+,.0f})")
        log.info(f"    Daily: {self.daily_r:+.2f}R (${self.daily_pnl_usd:+,.0f})")

        self._reset_trade()
        self._check_withdraw_threshold()

    def _close_trade(self, reason: str):
        t = self.trade
        if not t:
            return

        log.info(f"    Closing trade: {reason}")

        if t.pending:
            log.info(f"    Entry still pending — cancelling limit order")
            self.broker.cancel_order(t.order_ids.get('entry'))
            self._reset_trade()
            return

        self.broker.flatten()

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

        log.info(f"    Result: {total_r:+.2f}R (${pnl_usd:+,.0f}) | "
                 f"Daily: {self.daily_r:+.2f}R (${self.daily_pnl_usd:+,.0f})")

        self._reset_trade()

    def _check_withdraw_threshold(self):
        acct = self.broker.get_account_info()
        bal = acct.get('balance', self.start_balance)
        if bal > self.peak_balance:
            self.peak_balance = bal
        self._update_dd_floor()

        if self.winning_days < 5:
            return

        available = bal - (self.dd_floor + self.withdraw_buffer_usd)
        if available < self.min_withdraw_usd:
            return

        payout_amt = min(2000, int(available / 100) * 100)
        if payout_amt < self.min_withdraw_usd:
            return

        log.info(f"\n{'*'*55}")
        log.info(f"WITHDRAW READY — Balance ${bal:,.0f}")
        log.info(f"DD floor: ${self.dd_floor:,.0f} | Buffer: ${self.withdraw_buffer_usd:,.0f}")
        log.info(f"Available: ${available:,.0f} | Withdraw: ${payout_amt:,}")
        log.info(f"{'*'*55}\n")

    def shutdown(self):
        if self.trade:
            if self.trade.pending:
                log.info("Shutdown — cancelling pending entry")
                self.broker.cancel_order(self.trade.order_ids.get('entry'))
            else:
                log.info("Shutdown — flattening open position")
                self.broker.flatten()
            self._reset_trade()
        log.info("Executor shutdown complete.")
