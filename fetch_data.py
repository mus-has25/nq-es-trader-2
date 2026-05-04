#!/usr/bin/env python3
"""Fetch NQ/ES futures data from Yahoo Finance (no API key required)."""
import requests
import pandas as pd
import os
from datetime import datetime, timedelta, timezone

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
HEADERS = {'User-Agent': 'Mozilla/5.0'}


def fetch_yf(symbol: str, yf_ticker: str, interval: str, days: int) -> pd.DataFrame:
    """Fetch OHLCV bars from Yahoo Finance chart API."""
    print(f"Fetching {symbol} ({yf_ticker}) interval={interval} days={days}...")

    # Yahoo Finance: 1m max 7d, 2m/5m max 60d
    all_frames = []
    chunk_days = 60
    end = datetime.now(timezone.utc)

    remaining = days
    while remaining > 0:
        chunk = min(remaining, chunk_days)
        start = end - timedelta(days=chunk)
        params = {
            'interval': interval,
            'period1': int(start.timestamp()),
            'period2': int(end.timestamp()),
        }
        r = requests.get(
            f'https://query1.finance.yahoo.com/v8/finance/chart/{yf_ticker}',
            params=params, headers=HEADERS, timeout=15,
        )
        r.raise_for_status()
        result = (r.json().get('chart') or {}).get('result') or []
        if not result:
            break

        timestamps = result[0].get('timestamp', [])
        q = result[0].get('indicators', {}).get('quote', [{}])[0]
        if not timestamps:
            end = start
            remaining -= chunk
            continue

        df = pd.DataFrame({
            'datetime': pd.to_datetime(timestamps, unit='s', utc=True),
            'open':   q.get('open',   []),
            'high':   q.get('high',   []),
            'low':    q.get('low',    []),
            'close':  q.get('close',  []),
            'volume': q.get('volume', []),
        })
        df = df.dropna(subset=['open', 'close'])
        all_frames.append(df)
        end = start
        remaining -= chunk

    if not all_frames:
        raise RuntimeError(f"No data returned for {yf_ticker}")

    df = pd.concat(all_frames, ignore_index=True)
    df['datetime'] = df['datetime'].dt.tz_convert('US/Eastern').dt.tz_localize(None)
    df = df.sort_values('datetime').drop_duplicates('datetime').reset_index(drop=True)
    df['symbol'] = symbol
    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 2-min intraday bars (~38 calendar days)
    for symbol, ticker in [('NQ', 'NQ=F'), ('ES', 'ES=F')]:
        df = fetch_yf(symbol, ticker, interval='2m', days=60)
        path = os.path.join(DATA_DIR, f'{symbol}_2min.csv')
        df.to_csv(path, index=False)
        print(f"  {len(df):,} bars → {path}")
        print(f"  Range: {df['datetime'].min()} → {df['datetime'].max()}")

    # Daily bars going back 2 years — for regime detection warmup
    for symbol, ticker in [('NQ', 'NQ=F'), ('ES', 'ES=F')]:
        df = fetch_yf(symbol, ticker, interval='1d', days=730)
        path = os.path.join(DATA_DIR, f'{symbol}_daily.csv')
        df.to_csv(path, index=False)
        print(f"  {len(df):,} daily bars → {path}")
        print(f"  Range: {df['datetime'].min()} → {df['datetime'].max()}")

    print("\nDone.")


if __name__ == '__main__':
    main()
