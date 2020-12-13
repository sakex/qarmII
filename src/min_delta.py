import numpy as np
from scipy.optimize import minimize

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

    markowitz_returns_returns = mean_variance.returns()
    markowitz_final_avg = np.mean(markowitz_returns_returns)

    success = False
    res = None
    while not success:
        initial_weights = np.random.random(tickers)
        # , bounds=[(0, 1) for _ in range(len(initial_weights))]
        res = minimize(beta, initial_weights)
        found_weights = res.x
        weight_portfolio = WeightedPortfolio(price_matrix, found_weights / np.sum(found_weights))
        end_portfolio_returns = weight_portfolio.returns()
        average_score = np.mean(end_portfolio_returns)
        print("Nu portfolio", average_score, "Nu Market", markowitz_final_avg, "beta",
              beta(found_weights / np.sum(found_weights)))
        success = (res.fun <= 0.01) and average_score >= markowitz_final_avg

    end_weights = res.x
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
