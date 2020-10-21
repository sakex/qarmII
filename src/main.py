import aiohttp
import asyncio

from data_service.fetch_binance import fetch_many_tickers


async def main():
    async with aiohttp.ClientSession() as session:
        candles = await fetch_many_tickers(session, ["BTCUSDT", "ETHUSDT"], 10, "1h")
        print(candles)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
