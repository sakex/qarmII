__author__ = "Alexandre Senges"
__copyright__ = "Copyright (C) 2020 Author Name"
__license__ = "Public Domain"
__version__ = "1.0"

from dataclasses import dataclass


@dataclass
class BinanceCandle:
    close_time: int
    close: float
    volume: float
