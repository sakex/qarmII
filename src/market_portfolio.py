from abc import ABC, abstractmethod
import numpy as np


class MarketPortfolio(ABC):
    _stock_returns: np.ndarray
    __weights: np.ndarray
    __returns: np.ndarray

    def __init__(self, stock_returns: np.ndarray):
        self._stock_returns = stock_returns
        self.__weights = self.generate_portfolio()
        self.__returns = self.__weights.T.dot(self._stock_returns)

    def returns(self):
        return self.__returns

    def weights(self):
        return self.__weights

    @abstractmethod
    def generate_portfolio(self) -> np.ndarray:
        pass


class EqualyWeightedPortfolio(MarketPortfolio):
    def generate_portfolio(self) -> np.ndarray:
        tickers = self._stock_returns.shape[0]
        return np.full((tickers, 1), 1 / self._stock_returns.shape[1])
