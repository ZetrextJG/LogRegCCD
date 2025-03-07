import numpy as np
from tqdm import tqdm

from metrics import calculate_metrics, Metrics
from typing import TypedDict


def soft_thresholding(x: np.ndarray, gamma: float):
    """Soft thresholding operator (Equation 6)"""
    inner = np.abs(x) - gamma
    return np.sign(x) * np.maximum(inner, 0)


def sigmoid(x: np.ndarray, float_upper_bound: float = 600.0):
    # Prevents overflow
    x = np.clip(x, a_min=-float_upper_bound, a_max=float_upper_bound)
    return 1 / (1 + np.exp(-x))


class Result(TypedDict):
    lmbda: float
    beta0: float
    betas: np.ndarray
    metrics: Metrics


class LogRegCCD:
    beta0: float
    betas: np.ndarray
    lmbda: float

    def __init__(
        self,
        alpha: float,
        num_lmbdas: int = 100,
        min_lmbda_eps: float = 1e-4,
        max_cycles: int = 100,
        warm_start: bool = True,
        px_clipping_eps: float = 1e-5,
        heuristic_intercept: bool = False,
        fit_intercept: bool = False,
    ) -> None:
        self.alpha = alpha
        self.num_lmbdas = num_lmbdas
        self.min_lmbda_eps = min_lmbda_eps
        self.max_cycles = max_cycles
        self.warm_start = warm_start
        self.px_clipping_eps = px_clipping_eps
        self.heuristic_intercept = heuristic_intercept
        self.fit_intercept = fit_intercept

        self.fitted = False
        self.beta0 = None  # type: ignore
        self.betas = None  # type: ignore
        self.lmbda = None  # type: ignore

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._predict_proba(x, self.beta0, self.betas)

    def _predict_proba(
        self, x: np.ndarray, beta0: float, betas: np.ndarray
    ) -> np.ndarray:
        """
        Predicts the probabilities of the positive class
        using the current estimated of the coefficients.
        (Equation 11 using y as the I(G=1))
        """
        z_pred = beta0 + (x @ betas)
        return sigmoid(z_pred)

    def _fit(self, x: np.ndarray, y: np.ndarray, lmbda: float):
        # Precomputed for efficiency
        x2 = x**2

        # Initialize the betas
        if self.warm_start and (self.beta0 is not None) and (self.betas is not None):
            betas = self.betas.copy()
            beta0 = self.beta0
        else:
            betas = np.zeros(x.shape[1])
            beta0 = 0
            if self.heuristic_intercept:
                beta0 = -np.log((1 / np.mean(y) - 1))  # type: ignore

        # CCD loop
        for _ in range(self.max_cycles):
            old_betas = betas.copy()

            # Estimate using the current betas
            z_pred = beta0 + (x @ betas)  # linear predictor
            px_pred = sigmoid(z_pred)  # probability of the positive class
            ## Prediction clipping
            px_pred = np.clip(
                px_pred, a_min=self.px_clipping_eps, a_max=1 - self.px_clipping_eps
            )
            weights = px_pred * (1 - px_pred)  # weights (Equation 17)
            z = z_pred + (y - px_pred) / weights  # working response (Equation 16)

            if self.fit_intercept:
                beta0 = np.mean(z)

            # Run the coordinate descent
            weighted_vars = weights @ x2
            for j in range(x.shape[1]):
                z_pred = beta0 + (x @ betas)
                z_residuals = z - z_pred

                # Compute the update for beta_j
                ## Using Equation 8 adapted for weighted non standardized case
                s_input = np.sum(
                    weights * (x[:, j] * z_residuals + x2[:, j] * betas[j])
                )
                ## Update beta_j using Equation 10
                numerator = soft_thresholding(s_input, lmbda * self.alpha)
                demoniator = weighted_vars[j] + lmbda * (1 - self.alpha)
                betas[j] = numerator / demoniator

            if np.linalg.norm(betas - old_betas, ord=1) < 1e-4:
                break

        return beta0, betas

    def fit(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        val_X: np.ndarray,
        val_y: np.ndarray,
        metric: str = "accuracy",
    ):
        """X is of shape (n, p) and y is of shape (n,)"""
        assert (
            train_X.shape[0] == train_y.shape[0]
        ), "X and y should have the same number of samples"
        assert train_y.ndim == 1, "Labels should be 1D"
        assert np.all(np.isin(train_y, [0, 1])), "Labels should be binary"

        if self.fitted:
            raise ValueError("Model is already fitted")

        # Parameter reassignment to follow the notation in the paper
        x = train_X
        y = train_y
        N = x.shape[0]

        # Compute the lambda sequence (Section 2.5)
        # The scaling by 4 is due to max of weights being 1/4
        lmbda_max = 4 * np.max(np.abs(x.T @ y)) / (N * self.alpha)
        lmbda_min = self.min_lmbda_eps * lmbda_max
        lmbdas = np.logspace(np.log10(lmbda_max), np.log10(lmbda_min), self.num_lmbdas)

        # Fit the model for each lambda
        results: list[Result] = []
        for lmbda in tqdm(lmbdas):
            beta0, betas = self._fit(x, y, lmbda)
            probs = self._predict_proba(val_X, beta0, betas)
            metrics = calculate_metrics(val_y, probs)
            results.append(
                {
                    "lmbda": lmbda,
                    "beta0": beta0,  # type: ignore
                    "betas": betas,
                    "metrics": metrics,
                }
            )

        # Select the best lambda
        scores = [result["metrics"][metric] for result in results]
        scores = np.array(scores)
        idx = np.argmax(scores)

        self.lmbda = results[idx]["lmbda"]
        self.beta0 = results[idx]["beta0"]
        self.betas = results[idx]["betas"]

        self.fitted = True

        return results

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predicts on the input X"""
        proba = self.predict_proba(X)
        return (proba > threshold).astype(int)

    def fit_predict(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        val_X: np.ndarray,
        val_y: np.ndarray,
    ) -> np.ndarray:
        """Fits the model on the train set and predicts on the test set"""
        self.fit(train_X, train_y, val_X, val_y)
        return self.predict(train_X)
