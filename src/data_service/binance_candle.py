from dataclasses import dataclass


@dataclass
class BinanceCandle:
    close_time: int
    close: float
    volume: float
