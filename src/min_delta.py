__author__ = "Alexandre Senges"
__copyright__ = "Copyright (C) 2020 Author Name"
__license__ = "Public Domain"
__version__ = "1.0"

import json
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis

from market_portfolio import WeightedPortfolio, MeanVariancePortfolio


def minimize_beta(price_matrix: np.ndarray) -> np.ndarray:
    mean_variance = MeanVariancePortfolio(price_matrix)
    market_returns = mean_variance.returns()
    tickers = price_matrix.shape[0]

    normalized = price_matrix.copy()
    for row in normalized:
        row /= row[0]

    def beta(weights: np.ndarray) -> float:
        portfolio_equity = weights.dot(normalized)
        returns = np.diff(portfolio_equity) / portfolio_equity[:-1]
        return np.abs(np.corrcoef(returns, market_returns)[0, 1])

    print(mean_variance.portfolio_value())
    market_final_score = mean_variance.portfolio_value()[-1][-1]
    # markowitz_returns_returns = mean_variance.returns()
    # markowitz_final_avg = np.mean(markowitz_returns_returns)

    success = False
    res = None
    while not success:
        initial_weights = np.random.random(tickers)
        res = minimize(beta, initial_weights)
        found_weights = res.x
        weight_portfolio = WeightedPortfolio(price_matrix, (found_weights / np.sum(found_weights) * .999))
        final_score = weight_portfolio.portfolio_value()[-1][-1]
        b = beta(found_weights / np.sum(found_weights))
        print("Result portfolio", final_score, "Market Final Score", market_final_score, "beta", b)
        success = (-.1 <= b <= .1) and final_score >= market_final_score

    end_weights = res.x
    print("FINAL WEIGHT", json.dumps(list(end_weights)))
    return end_weights


def betting_on_beta(price_matrix: np.ndarray):
    mean_variance = MeanVariancePortfolio(price_matrix)
    market_returns = mean_variance.returns()
    returns = np.diff(price_matrix) / price_matrix[:, :-1]
    betas = [(np.corrcoef(row, market_returns)[0, 1], index) for index, row in enumerate(returns)]
    betas.sort(key=lambda t: t[0])
    median_pos = len(betas) >> 1
    median = betas[median_pos][0]
    normalized = price_matrix.copy()
    for row in normalized:
        row /= row[0]
    weighted_portfolio = np.array([normalized[beta[1]] * (median - beta[0]) for beta in betas])
    print(np.sum(weighted_portfolio, axis=0))


def pick_best_ratio(price_matrix: np.ndarray) -> (List[int], np.ndarray):
    mean_variance = MeanVariancePortfolio(price_matrix)
    market_returns = mean_variance.returns()
    returns = np.diff(price_matrix) / price_matrix[:, :-1]

    betas = np.array([np.corrcoef(row, market_returns)[0, 1] for row in returns])
    means = np.mean(returns, axis=1)
    plt.plot(betas, means, 'o')

    model = LinearRegression()
    model.fit(betas.reshape(-1, 1), means.reshape(-1, 1))

    x_axis = np.linspace(np.min(betas), np.max(betas), len(betas))
    y_line = model.predict(x_axis[:, np.newaxis])
    plt.plot(x_axis, y_line)
    plt.title("Linear regression Betas vs Mu")
    plt.xlabel("Beta")
    plt.ylabel("Mu")
    plt.show()

    kept_indices = [index for index, (real, predicted) in enumerate(zip(means, y_line)) if real >= predicted]
    new_returns_matrix = np.array([returns[index] for index in kept_indices])

    def mean_returns(weights: np.array) -> float:
        r = weights.dot(new_returns_matrix)
        return -np.mean(r) / np.std(r)

    initial_weights = np.full(len(kept_indices), 1 / len(kept_indices))

    def sum_constraint(weights: np.ndarray):
        return np.sum(weights) - 1

    res = minimize(mean_returns, initial_weights, bounds=[(0, 1) for _ in range(len(kept_indices))],
                   constraints={'type': 'eq', 'fun': sum_constraint})
    optimized_weights = res.x
    return kept_indices, optimized_weights


def max_dd(prices: np.ndarray) -> float:
    current_max = -np.inf
    dd = np.inf
    for price in prices[0]:
        if price > current_max:
            current_max = price
        elif (price - current_max) / current_max < dd:
            dd = (price - current_max) / current_max
    return dd


def print_betas(price_matrix: np.ndarray):
    equaly_weighted = MeanVariancePortfolio(price_matrix)
    returns = np.diff(price_matrix) / price_matrix[:, :-1]
    market_returns = equaly_weighted.returns()

    betas = [np.corrcoef(r, market_returns)[0, 1] for r in returns]
    print(min(betas), max(betas))


def show_stats(kept_indices: List[int], price_matrix: np.ndarray, optimized_weights: np.ndarray,
               ticker_names: List[str]):
    mean_variance = MeanVariancePortfolio(price_matrix)
    market_returns = mean_variance.returns()
    returns = np.diff(price_matrix) / price_matrix[:, :-1]
    new_returns_matrix = np.array([returns[index] for index in kept_indices])
    new_price_matrix = np.array([price_matrix[index] for index in kept_indices])
    optimized_mu = np.mean(optimized_weights.dot(new_returns_matrix))

    print("Optimized mu", optimized_mu)
    print("Market mu", np.mean(market_returns))
    print(optimized_weights)
    portfolio = WeightedPortfolio(new_price_matrix, optimized_weights)
    print("Sharpe Ratio", portfolio.sharpe_ratio())
    print("Volatility", portfolio.volatility())
    beta = np.corrcoef(portfolio.returns(), market_returns)[0, 1]
    print("Beta", beta)
    print("Skewness", skew(portfolio.returns(), axis=1))
    print("Kurtosis", kurtosis(portfolio.returns(), axis=1))
    mark_ret = mean_variance.portfolio_value()[-1, -1]
    print("Market returns", mark_ret)
    p_ret = portfolio.portfolio_value()[-1, -1]
    print("Portfolio returns", p_ret)
    print("Alpha", p_ret - (.02 * beta * (mark_ret - .02)))
    print("Information ratio", (p_ret - mark_ret) / np.std(market_returns - portfolio.returns()))
    print("Max drowdown", max_dd(portfolio.portfolio_value()))

    x = np.arange(0, len(price_matrix[0]))

    normalized = price_matrix.copy()
    for row in normalized:
        row /= row[0]

    for i, l in zip(normalized, ticker_names):
        plt.plot(x, i)
        # plt.legend(labels)
    plt.title("Crypto graph")
    plt.plot(x, portfolio.portfolio_value()[0], 'black', label='Portfolio')
    plt.legend('Portfolio')
    leg = plt.legend()
    leg.get_lines()[-1].set_linewidth(4)
    plt.show()
