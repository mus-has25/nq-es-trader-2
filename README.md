# NQ Trading Bot - TopStepX 50K Funded Account

Autonomous MNQ futures trading bot for TopStepX funded accounts. Runs 2 models (OU Reversion, VWAP Reversion) with a 2-phase schedule and manages all entries, exits, and risk automatically.

## Quick Start

### 1. Install Python 3.10+

- **Mac**: `brew install python` or download from [python.org](https://www.python.org/downloads/)
- **Windows**: Download from [python.org](https://www.python.org/downloads/) (check "Add to PATH" during install)

### 2. Install dependencies

```bash
cd nq-es-trader
pip install -r requirements.txt
```

### 3. Set up credentials

Copy the example env file and fill in your TopStepX credentials:

```bash
cp .env.example .env
```

Edit `.env` with your details:
```
TOPSTEP_USER=your_topstep_username
TOPSTEP_API_KEY=your_api_key
TOPSTEP_ENV=demo
CONTRACTS=20
```

Get your API key from: TopStepX dashboard > API Access

### 4. Run the bot

**Mac**: Double-click `start_bot.command`

**Windows**: Double-click `start_bot.bat`

**Or from terminal**:
```bash
python run_live.py
```

Press `Ctrl+C` to stop and flatten all positions.

## Funded Account Rules

| Rule | Value |
|------|-------|
| Starting balance | $50,000 |
| Trailing drawdown | $2,000 (trails until peak $52K, then locks floor at $50K) |
| Max position | 20 MNQ (2 NQ) |
| Profit split | 90/10 |
| Max payout | $2,000 per withdrawal |
| First withdrawal | $1K at $53K balance (Phase 1, $2K buffer) |
| Subsequent withdrawals | $2K at $54K balance (Phase 2, $2K buffer) |
| Payout eligibility | 5 winning trading days |

## Strategy

- **OU Reversion**: Mean-reversion on Ornstein-Uhlenbeck process (PF 3.10, 131 trades)
- **VWAP Reversion**: VWAP-based mean reversion (PF 1.83, 197 trades)

Overall (filtered): 67.7% win rate, 2.25 profit factor, 328 trades at 20 MNQ.
Signals with 30-50 tick risk are skipped (55% WR noise).

**Phase 1** (until first payout): Mon-Fri mornings, 20 MNQ. Withdraw $1K at $53K.
**Phase 2** (after first payout): Mon-Fri mornings, 20 MNQ. Withdraw $2K at $54K.

## Risk Management

- $2,000 trailing drawdown (locks at $50K floor when peak hits $52K)
- Skip 30-50 tick risk signals (dead zone filter)
- Reduce to 15 MNQ when DD >= $1,500
- $2,000 withdrawal buffer above DD floor
- Phase 1: withdraw $1K at $53K, Phase 2: withdraw $2K at $54K
- Breakeven stops, partial exits
- Time stops per model (30-45 min)
- Auto-flatten at 2:55 PM CT
- Daily loss cap after first loss (-0.25R)
