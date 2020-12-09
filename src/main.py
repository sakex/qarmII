from typing import Dict, List

import aiohttp
import asyncio
import numpy as np

from data_service.fetch_binance import fetch_many_tickers
from market_portfolio import EqualyWeightedPortfolio
from src.data_service.binance_candle import BinanceCandle

TICKERS = ["BTC", "ETH", "XRP", "LTC", "LINK", "QTUM", "ICX",
           "BCH", "IOTA", "DASH", "NEO", "HBAR", "NANO",
           "ADA", "BNB", "EOS", "ZIL", "ALGO", "BAT", "BTT", "ENJ",
           "TRX", "XTZ", "ATOM", "ZEC", "ETC", "THETA", "ONT", "OMG",
           "DOGE"]  # For the meme


async def main():
    async with aiohttp.ClientSession() as session:
        pairs: List[str] = [f"{t}USDT" for t in TICKERS]
        candles: Dict[str, List[BinanceCandle]] = await fetch_many_tickers(session, pairs, 10, "1h")
        price_matrix: np.ndarray = np.array([[candle.close for candle in row] for row in candles.values()], dtype=float)
        return_matrix = np.diff(np.log(price_matrix))
        equaly_weighted = EqualyWeightedPortfolio(return_matrix)
        print(equaly_weighted.returns())


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
