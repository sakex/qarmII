from abc import ABC, abstractmethod
import numpy as np


class MarketPortfolio(ABC):
    _prices: np.ndarray
    __weights: np.ndarray
    __returns: np.ndarray
    __portfolio_value: np.ndarray

    def __init__(self, _prices: np.ndarray):
        self._prices = _prices
        self.__weights = self.generate_portfolio()
        normalized = self._prices.copy()
        for row in normalized:
            row /= row[0]
        self.__portfolio_value = self.__weights.dot(normalized)  # 1d
        self.__returns = np.diff(self.__portfolio_value) / self.__portfolio_value[:, :-1]

    def returns(self) -> np.ndarray:
        return self.__returns

    def portfolio_value(self) -> np.ndarray:
        return self.__portfolio_value

    def weights(self) -> np.ndarray:
        return self.__weights

    def prices_normalized(self) -> np.ndarray:
        return self._prices

    def volatility(self) -> float:
        return np.sqrt(365 * 24) * np.std(self.__returns)

    def sharpe_ratio(self):
        return np.mean(self.__returns) * 365 * 24 / self.volatility()

    @abstractmethod
    def generate_portfolio(self) -> np.ndarray:
        pass


class EqualyWeightedPortfolio(MarketPortfolio):
    def generate_portfolio(self) -> np.ndarray:
        tickers = self._prices.shape[0]
        return np.full((1, tickers), 1 / tickers)


class WeightedPortfolio(MarketPortfolio):
    def __init__(self, _prices: np.ndarray, predefined_weights: np.ndarray):
        self.predefined_weights = predefined_weights.reshape(1, _prices.shape[0])
        super().__init__(_prices)

    def generate_portfolio(self) -> np.ndarray:
        return self.predefined_weights


class MeanVariancePortfolio(MarketPortfolio):
    def generate_portfolio(self) -> np.ndarray:
        returns = np.diff(self._prices) / self._prices[:, :-1]
        means = np.mean(returns, axis=1)
        cov_matrix = np.cov(returns)
        weights = np.linalg.inv(cov_matrix).dot(means)
        return (weights / np.sum(weights)).reshape(1, cov_matrix.shape[0])
