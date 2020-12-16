from typing import Dict, List

import aiohttp
import asyncio
import numpy as np

from data_service.fetch_binance import fetch_many_tickers
from market_portfolio import EqualyWeightedPortfolio, WeightedPortfolio, MeanVariancePortfolio
from min_delta import minimize_beta, betting_on_beta
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
        price_matrix: np.ndarray = np.array([[candle.close for candle in row] for row in candles.values()], dtype=float)
        betting_on_beta(price_matrix)
        weights = minimize_beta(price_matrix)
        weight_portfolio = WeightedPortfolio(price_matrix, (weights / np.sum(weights) * .999))
        print("Weighted portfolio", weight_portfolio.portfolio_value())
        print("Variance", weight_portfolio.variance())
        print("Sharpe ratio", weight_portfolio.sharpe_ratio())


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
