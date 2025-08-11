\"\"\" 
Binance Multi-Pair 13x Compounding Futures Bot (MAINNET READY)
File: binance_multi_pair_compound_bot.py

WARNING - READ CAREFULLY:
- This script is educational. Trading with 13x leverage is extremely high-risk and can wipe your account quickly.
- DO NOT run this with large funds without extensive testing and monitoring.
- The author is not responsible for any losses.

FEATURES:
- Scans multiple Binance USDT-M futures pairs (configurable via TRADING_PAIRS env var).
- Picks the strongest breakout candidate (momentum + volume) and trades it.
- Uses configurable leverage (default 13x), stop-loss, take-profit, and risk per trade.
- Compounds position sizing based on current wallet USDT balance.
- Includes a keep-alive Flask server for Railway deployments.
- Basic safety circuit-breakers: max daily loss and max consecutive losses.

USAGE (Railway):
- Set the following environment variables (Railway project > Variables):
  BINANCE_API_KEY (DO NOT COMMIT), BINANCE_API_SECRET (DO NOT COMMIT),
  TRADING_PAIRS (comma-separated, e.g. BTCUSDT,ETHUSDT,PEPEUSDT),
  LEVERAGE (13), RISK_PER_TRADE (0.05), STOPLOSS_PCT (0.015), TAKE_PROFIT_PCT (0.06),
  TIMEFRAME (1m), MIN_BALANCE (10), MAX_DAILY_LOSS (0.2), MAX_CONSECUTIVE_LOSSES (5), POLL_INTERVAL (5)

DEPENDENCIES: ccxt, pandas, numpy, python-dotenv, flask
\"\"\"

import os
import time
import math
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple

import ccxt
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Keep-alive server for Railway
from flask import Flask
from threading import Thread

# ----- Logging -----
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ----- Load env -----
load_dotenv()
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')
PAIRS_RAW = os.getenv('TRADING_PAIRS', 'PEPEUSDT')
TRADING_PAIRS: List[str] = [p.strip().upper() for p in PAIRS_RAW.split(',') if p.strip()]
LEVERAGE = int(os.getenv('LEVERAGE', '13'))
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.05'))
STOPLOSS_PCT = float(os.getenv('STOPLOSS_PCT', '0.015'))
TAKE_PROFIT_PCT = float(os.getenv('TAKE_PROFIT_PCT', '0.06'))
TIMEFRAME = os.getenv('TIMEFRAME', '1m')
MIN_BALANCE = float(os.getenv('MIN_BALANCE', '10'))
MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', '0.2'))  # 20% of starting balance
MAX_CONSECUTIVE_LOSSES = int(os.getenv('MAX_CONSECUTIVE_LOSSES', '5'))
POLL_INTERVAL = int(os.getenv('POLL_INTERVAL', '5'))

# ----- Keep-alive (Flask) -----
app = Flask('')

@app.route('/')
def home():
    return 'Multi-pair bot alive'

def run_web():
    try:
        app.run(host='0.0.0.0', port=8080)
    except Exception as e:
        logger.warning('Keep-alive server error: %s', e)

def keep_alive():
    t = Thread(target=run_web)
    t.daemon = True
    t.start()

# ----- Dataclasses -----
@dataclass
class Position:
    symbol: str
    side: str
    entry_price: float
    size_usdt: float
    stop_price: float
    take_price: float
    qty: float
    order_id: Optional[str] = None

# ----- Exchange init -----
def create_exchange():
    if not API_KEY or not API_SECRET:
        raise ValueError('API_KEY and API_SECRET must be set as environment variables')

    exchange = ccxt.binanceusdm({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',
        }
    })
    return exchange

# ----- Market data & signals -----
def fetch_ohlcv(exchange, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    return df

def breakout_score(df: pd.DataFrame, N: int = 20) -> float:
    # Score based on how much last close is above rolling max and volume spike
    df = df.copy()
    df['rolling_max'] = df['close'].rolling(N).max()
    df['vol_mean'] = df['vol'].rolling(N).mean()
    latest = df.iloc[-1]
    if pd.isna(latest['rolling_max']) or pd.isna(latest['vol_mean']):
        return 0.0
    price_gap = (latest['close'] - latest['rolling_max']) / (latest['rolling_max'] + 1e-12)
    vol_ratio = (latest['vol'] / (latest['vol_mean'] + 1e-12)) - 1.0
    score = max(0.0, price_gap) * 0.7 + max(0.0, vol_ratio) * 0.3
    return float(score)

# ----- Risk sizing -----
def get_balance(exchange) -> float:
    bal = exchange.fetch_balance(params={})
    usdt = 0.0
    try:
        usdt = float(bal['total']['USDT'])
    except Exception:
        try:
            usdt = float(bal['USDT']['total'])
        except Exception:
            for k, v in bal.items():
                if isinstance(v, dict) and 'USDT' in v:
                    try:
                        usdt = float(v['USDT']['total'])
                        break
                    except Exception:
                        continue
    return usdt

def compute_position_size(balance_usdt: float, stoploss_pct: float, risk_fraction: float, leverage: int) -> Tuple[float, float]:
    risk_amount = balance_usdt * risk_fraction
    if stoploss_pct <= 0:
        raise ValueError('stoploss_pct must be > 0')
    raw_size = risk_amount / stoploss_pct
    max_size = balance_usdt * leverage * 0.99
    size_usdt = min(raw_size, max_size)
    margin_required = size_usdt / leverage
    return size_usdt, margin_required

# ----- Order helpers -----
def set_symbol_leverage(exchange, symbol: str, leverage: int):
    try:
        exchange.fapiPrivate_post_leverage({'symbol': symbol.replace('/',''), 'leverage': int(leverage)})
        logger.info('Leverage set to %s for %s', leverage, symbol)
    except Exception as e:
        logger.warning('Could not set leverage via endpoint for %s: %s', symbol, e)

def truncate_qty(exchange, symbol: str, qty: float) -> float:
    try:
        markets = exchange.load_markets()
        info = markets.get(symbol)
        if not info:
            return qty
        precision = info.get('precision', {}).get('amount')
        if precision is None:
            return qty
        factor = 10 ** precision
        truncated = math.floor(qty * factor) / factor
        return truncated
    except Exception:
        return qty

def place_market_buy(exchange, symbol: str, size_usdt: float) -> Tuple[dict, float, float]:
    ticker = exchange.fetch_ticker(symbol)
    price = float(ticker['last'])
    qty = size_usdt / price
    qty = truncate_qty(exchange, symbol, qty)
    if qty <= 0:
        raise ValueError('qty computed <= 0')
    order = exchange.create_market_buy_order(symbol, qty, {})
    return order, price, qty

def close_position_market_sell(exchange, symbol: str, qty: float):
    qty = truncate_qty(exchange, symbol, qty)
    return exchange.create_market_sell_order(symbol, qty, {})

# ----- Main bot loop -----
def run_bot():
    keep_alive()
    exchange = create_exchange()
    markets = exchange.load_markets()

    # Ensure provided pairs exist
    valid_pairs = [p for p in TRADING_PAIRS if p in markets]
    if not valid_pairs:
        raise ValueError('None of the TRADING_PAIRS are available on the exchange: ' + ','.join(TRADING_PAIRS))
    logger.info('Trading pairs: %s', valid_pairs)

    # set leverage for each pair (best-effort)
    for p in valid_pairs:
        try:
            set_symbol_leverage(exchange, p, LEVERAGE)
        except Exception:
            pass

    starting_balance = get_balance(exchange)
    logger.info('Starting wallet USDT: %.4f', starting_balance)
    daily_loss_limit = starting_balance * MAX_DAILY_LOSS
    consec_losses = 0

    open_position: Optional[Position] = None

    while True:
        try:
            balance = get_balance(exchange)
            logger.info('Wallet USDT: %.4f', balance)

            if balance < MIN_BALANCE:
                logger.warning('Balance %.2f lower than MIN_BALANCE %.2f. Sleeping.', balance, MIN_BALANCE)
                time.sleep(30)
                continue

            # Check daily loss
            if starting_balance - balance >= daily_loss_limit:
                logger.warning('Daily loss limit reached. Stopping trading until manual restart.')
                break

            # If no open position, scan pairs for best breakout score
            if open_position is None:
                best_score = 0.0
                best_pair = None

                for pair in valid_pairs:
                    try:
                        df = fetch_ohlcv(exchange, pair, TIMEFRAME, limit=200)
                        score = breakout_score(df)
                        logger.debug('Pair %s score %.6f', pair, score)
                        if score > best_score:
                            best_score = score
                            best_pair = pair
                    except Exception as e:
                        logger.warning('Error fetching %s: %s', pair, e)

                logger.info('Best pair: %s score=%.6f', best_pair, best_score)

                # threshold to avoid tiny noise trades
                SCORE_THRESHOLD = 0.0005
                if best_pair and best_score >= SCORE_THRESHOLD:
                    size_usdt, margin_required = compute_position_size(balance, STOPLOSS_PCT, RISK_PER_TRADE, LEVERAGE)
                    logger.info('Preparing trade on %s - size_usdt=%.2f margin_req=%.2f', best_pair, size_usdt, margin_required)

                    if margin_required > balance * 0.98:
                        logger.warning('Not enough margin for %s. Skipping.', best_pair)
                    else:
                        # open market buy
                        order, entry_price, qty = place_market_buy(exchange, best_pair, size_usdt)
                        stop_price = entry_price * (1 - STOPLOSS_PCT)
                        take_price = entry_price * (1 + TAKE_PROFIT_PCT)
                        open_position = Position(symbol=best_pair, side='long', entry_price=entry_price, size_usdt=size_usdt, stop_price=stop_price, take_price=take_price, qty=qty, order_id=order.get('id'))
                        logger.info('Opened LONG %s @%.8f size_usdt=%.2f qty=%.6f stop=%.8f take=%.8f', best_pair, entry_price, size_usdt, qty, stop_price, take_price)

            # Monitor open position
            if open_position is not None:
                ticker = exchange.fetch_ticker(open_position.symbol)
                last_price = float(ticker['last'])

                if last_price <= open_position.stop_price:
                    logger.info('Stop hit for %s: last %.8f <= stop %.8f. Closing.', open_position.symbol, last_price, open_position.stop_price)
                    try:
                        close_position_market_sell(exchange, open_position.symbol, open_position.qty)
                    except Exception as e:
                        logger.exception('Error closing on stop: %s', e)
                    open_position = None
                    consec_losses += 1
                elif last_price >= open_position.take_price:
                    logger.info('Take profit hit for %s: last %.8f >= take %.8f. Closing.', open_position.symbol, last_price, open_position.take_price)
                    try:
                        close_position_market_sell(exchange, open_position.symbol, open_position.qty)
                    except Exception as e:
                        logger.exception('Error closing on take profit: %s', e)
                    open_position = None
                    consec_losses = 0

                # safety: close if consecutive losses exceed limit
                if consec_losses >= MAX_CONSECUTIVE_LOSSES:
                    logger.warning('Max consecutive losses reached (%d). Stopping trading.', consec_losses)
                    break

            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            logger.info('Interrupted by user, exiting')
            break
        except Exception as e:
            logger.exception('Error in main loop: %s', e)
            time.sleep(5)

if __name__ == '__main__':
    run_bot()
