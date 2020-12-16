import asyncio
import aiohttp
from typing import List, Dict, Set
from datetime import datetime, timedelta
from functools import reduce

from data_service.binance_candle import BinanceCandle


def get_duration_period(period: str) -> timedelta:
    """
    Create time delta from string (so it's easier to work with)
    :param period: str - Has to be 15m/30m/1h/4h/1d
    :return: timedelta
    """
    if period == "15m":
        return timedelta(minutes=15)
    elif period == "30m":
        return timedelta(minutes=30)
    elif period == "1h":
        return timedelta(hours=1)
    elif period == "4h":
        return timedelta(hours=4)
    elif period == "1d":
        return timedelta(days=1)
    raise ValueError(f"Invalid period: {period}")


async def fetch(session: aiohttp.ClientSession, url: str) -> List[BinanceCandle]:
    """
    Send one fetch query
    :param session: Running current session
    :param url: Url to query
    :return: The raw list from the query
    """
    async with session.get(url) as response:
        return await response.json()


def exclude_duplicates(candles: List[BinanceCandle]) -> List[BinanceCandle]:
    """
    We don't trust our function at 100%, so we exclude any possible duplicate
    :param candles: The list of fetched candles
    :return: A list of candles without duplicates
    """
    close_time_set = set()
    remaining = []
    for candle in candles:
        if candle.close_time not in close_time_set:
            remaining.append(candle)
            close_time_set.add(candle.close_time)
    return remaining


def match_datasets(datasets) -> List[List[BinanceCandle]]:
    """
    Match de datasets and remove any candle that is not present in all the datasets
    :param datasets: List of datasets to match
    :return: Matched datasets
    """
    close_time_sets: List[Set[int]] = [set([candle.close_time for candle in dataset]) for dataset in datasets]
    intersection: Set[int] = reduce(lambda remaining, current: remaining.intersection(current),
                                    close_time_sets[1:], close_time_sets[0])
    return [[candle for candle in ds if candle.close_time in intersection] for ds in datasets]


async def fetch_one_ticker(session: aiohttp.ClientSession,
                           ticker: str,
                           queries_count: int,
                           delta: timedelta,
                           periods: str) -> List[BinanceCandle]:
    """
    Run many queries to fetch one ticker, then clean the data
    :param session: The running io session
    :param ticker: Ticker to be fetched (eg. BTCUSDT)
    :param queries_count: Number of queries to run
    :param delta: timedelta between two candles
    :param periods: str representation of the delta
    :return: Cleaned dataset for one candle
    """
    now = datetime.now()
    base_url = "https://api.binance.com/api/v3/klines?symbol="
    date_from = now - delta * queries_count * 1000
    d = delta * 1000
    queries_str = [
        f"{base_url}{ticker}&interval={periods}&limit=1000&startTime={int((date_from + df * d).timestamp() * 1000)}\
&endTime={int((date_from + (df + 1) * d).timestamp() * 1000)}"
        for df in range(queries_count)]
    queries = await asyncio.gather(*[fetch(session, query) for query in queries_str])
    candles = [BinanceCandle(c[6], float(c[4]), float(c[7])) for q in queries for c in q]
    candles = exclude_duplicates(candles)
    for c1, c2 in zip(candles[:-1], candles[1:]):
        assert c1.close_time < c2.close_time
    return candles


async def fetch_many_tickers(session: aiohttp.ClientSession,
                             tickers: List[str],
                             data_points: int,
                             periods: str) -> Dict[str, List[BinanceCandle]]:
    """
    Fetch data for many tickers
    :param session: running io session
    :param tickers: list of tickers, (eg. ["BTCUSDT", "ETHUSDT"])
    :param data_points: Number of queries per ticker
    :param periods: Number of sub periods
    :return: A dictionary which pairs the tickers with their data
    """
    delta = get_duration_period(periods)
    candles = await asyncio.gather(*[fetch_one_ticker(session, ticker, data_points, delta, periods)
                                     for ticker in tickers])
    candles = match_datasets(candles)
    return dict([(ticker, candle_list) for ticker, candle_list in zip(tickers, candles)])
