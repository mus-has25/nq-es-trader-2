#!/usr/bin/env python3
"""
NQ Multi-Model Trading Bot — TopStepX 50K
Connects directly to TopStepX via their REST API.

Usage:
  python run_live.py              # uses .env file
  python run_live.py --contracts 5 --env demo
"""
import argparse
import logging
import os
from dotenv import load_dotenv
from config import Config, InstrumentConfig
from live.broker_topstep import TopStepBroker
from live.executor_multi import LiveExecutor

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)


def main():
    p = argparse.ArgumentParser(description='NQ Trading Bot — TopStepX 50K')
    p.add_argument('--ou', type=int, default=20)
    p.add_argument('--trend', type=int, default=20)
    p.add_argument('--vwap', type=int, default=20)
    p.add_argument('--phase', default='eval', choices=['eval', 'xfa', 'payout'])
    p.add_argument('--env', default=os.getenv('TOPSTEP_ENV', 'demo'),
                   choices=['demo', 'live'])
    args = p.parse_args()

    username = os.getenv('TOPSTEP_USER')
    api_key = os.getenv('TOPSTEP_API_KEY')

    if not username or not api_key:
        print("ERROR: Missing credentials.")
        print("Copy .env.example to .env and fill in your TopStepX credentials:")
        print("  TOPSTEP_USER=your_username")
        print("  TOPSTEP_API_KEY=your_api_key")
        return

    cfg = Config()
    cfg.instrument = InstrumentConfig('MNQ', 0.25, 0.50, 2.0)
    cfg.account_size = 50000

    model_qty = {'ou_rev': args.ou, 'trend': args.trend, 'vwap_rev': args.vwap}

    phase_info = {
        'eval': '$3K target, $2K trailing DD, 50% consistency',
        'xfa':  '$0 start, $2K trailing DD, locks at $0 floor',
        'payout': '5 win days $150+, $2K max payout',
    }

    print(f"""
╔══════════════════════════════════════════════════════╗
║          NQ TRADING BOT — TOPSTEP 50K               ║
╠══════════════════════════════════════════════════════╣
║  Phase:      {args.phase.upper():<39}║
║  Sizing:     OU:{args.ou}  Trend:{args.trend}  VWAP:{args.vwap} MNQ{' ' * (22 - len(str(args.ou)) - len(str(args.trend)) - len(str(args.vwap)))}║
║  Mode:       {args.env.upper():<39}║
║  Models:     OU + Trend + VWAP Reversion             ║
║  Risk:       $500 max/trade, $600 daily cap          ║
║  Rules:      {phase_info[args.phase]:<39}║
╠══════════════════════════════════════════════════════╣
║  Press Ctrl+C to stop and flatten all positions      ║
╚══════════════════════════════════════════════════════╝
""")

    broker = TopStepBroker(username, api_key, env=args.env)

    try:
        broker.connect()
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Check your .env credentials and try again.")
        return

    executor = LiveExecutor(cfg, broker, model_qty=model_qty, phase=args.phase)

    try:
        executor.run()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        executor.shutdown()
        print("Clean exit.")


if __name__ == '__main__':
    main()
