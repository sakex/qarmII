__author__ = "Alexandre Senges"
__copyright__ = "Copyright (C) 2020 Author Name"
__license__ = "Public Domain"
__version__ = "1.0"

from typing import Dict, List

import aiohttp
import asyncio
import numpy as np

from data_service.fetch_binance import fetch_many_tickers
from market_portfolio import EqualyWeightedPortfolio, WeightedPortfolio, MeanVariancePortfolio
from min_delta import minimize_beta, betting_on_beta, pick_best_ratio, show_stats, print_betas
from src.data_service.binance_candle import BinanceCandle

TICKERS = ["BTC", "ETH", "XRP", "LTC", "LINK", "QTUM", "ICX",
           "BCH", "IOTA", "DASH", "NEO", "HBAR", "NANO",
           "ADA", "BNB", "EOS", "ZIL", "ALGO", "BAT", "BTT", "ENJ",
           "TRX", "XTZ", "ATOM", "ZEC", "ETC", "THETA", "ONT", "OMG",
           "DOGE"]  # For the meme


async def main():
    async with aiohttp.ClientSession() as session:
        pairs: List[str] = [f"{t}USDT" for t in TICKERS]
        candles: Dict[str, List[BinanceCandle]] = await fetch_many_tickers(session, pairs, 15, "1h")
        full_prices: np.ndarray = np.array([[candle.close for candle in row] for row in candles.values()], dtype=float)
        kept_indices_full, weights_full = pick_best_ratio(full_prices)
        show_stats(kept_indices_full, full_prices, weights_full, TICKERS)
        print_betas(full_prices)
        print("========")
        length = len(candles["ETHUSDT"])
        half = length >> 1
        print(f"Length: {length}, half: {half}")
        first_half_price_matrix: np.ndarray = np.array(
            [[candle.close for candle in row[:half]] for row in candles.values()], dtype=float)
        second_half_price_matrix: np.ndarray = np.array(
            [[candle.close for candle in row[half:]] for row in candles.values()], dtype=float)
        kept_indices, weights = pick_best_ratio(first_half_price_matrix)
        picked_tickers = [TICKERS[index] for index in kept_indices]
        print("Picked tickers:", picked_tickers)
        show_stats(kept_indices, first_half_price_matrix, weights, TICKERS)
        print("==============================SECOND HALF==============================")
        show_stats(kept_indices, second_half_price_matrix, weights, TICKERS)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
