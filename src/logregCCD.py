import numpy as np
from tqdm import tqdm

from metrics import calculate_metrics
from typing import TypedDict


def soft_thresholding(x: np.ndarray, gamma: float):
    """Soft thresholding operator (Equation 6)"""
    inner = np.abs(x) - gamma
    return np.sign(x) * np.maximum(inner, 0)


def sigmoid(x: np.ndarray, upper_bound: float = 1e2):
    # Prevents overflow
    x[x > upper_bound] = upper_bound
    x[x < -upper_bound] = -upper_bound
    return 1 / (1 + np.exp(-x))


class Result(TypedDict):
    lmbda: float
    score: float
    beta0: float
    betas: np.ndarray


class LogRegCCD:
    beta0: float
    betas: np.ndarray
    lmbda: float

    def __init__(
        self,
        alpha: float,
        num_lmbdas: int = 100,
        epsilon: float = 1e-3,
        max_cycles: int = 100,
    ) -> None:
        self.alpha = alpha
        self.num_lmbdas = num_lmbdas
        self.epsilon = epsilon
        self.max_cycles = max_cycles

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

    def _fit(
        self, x: np.ndarray, y: np.ndarray, lmbda: float, eps_clipping: float = 1e-5
    ):
        # Precomputed for efficiency
        N = x.shape[0]
        x2 = x**2

        # Initialize the betas
        betas = np.random.randn(x.shape[1])
        # beta0 = -np.log((1 / np.mean(y) - 1))  # type: ignore
        beta0 = 0

        # CCD loop
        for _ in range(self.max_cycles):
            old_betas = betas.copy()

            # Estimate using the current betas
            z_pred = beta0 + (x @ betas)  # linear predictor
            px_pred = sigmoid(z_pred)  # probability of the positive class
            ## Prediction clipping
            px_pred[px_pred < eps_clipping] = eps_clipping
            px_pred[px_pred > 1 - eps_clipping] = 1 - eps_clipping
            weights = px_pred * (1 - px_pred)  # weights (Equation 17)
            z = z_pred + (y - px_pred) / weights  # working response (Equation 16)

            # Run the coordinate descent
            for j in range(x.shape[1]):
                if betas[j] == 0:
                    continue

                z_pred = beta0 + (x @ betas)
                z_residuals = z - z_pred

                # Compute the update for beta_j
                ## Using Equation 8 adapted for weighted case
                s_input = np.mean(
                    weights * (x[:, j] * z_residuals + x2[:, j] * betas[j])
                )
                weighted_var = np.mean(weights * x2[:, j])
                ## Update beta_j using Equation 10
                numerator = soft_thresholding(s_input, lmbda * self.alpha)
                demoniator = weighted_var + lmbda * (1 - self.alpha)
                betas[j] = numerator / demoniator

            if np.linalg.norm(betas - old_betas) < (1e-3 / N):
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
        lmbda_max = np.max(np.abs(x.T @ y)) / (N * self.alpha)
        lmbda_min = self.epsilon * lmbda_max
        lmbdas = np.logspace(np.log10(lmbda_max), np.log10(lmbda_min), self.num_lmbdas)

        # Fit the model for each lambda
        results = []
        for lmbda in tqdm(lmbdas):
            beta0, betas = self._fit(x, y, lmbda)

            probs = self._predict_proba(val_X, beta0, betas)
            y_pred = (probs > 0.5).astype(int)
            scores = calculate_metrics(val_y, y_pred)
            score = scores[metric]

            results.append(
                {"lmbda": lmbda, "score": score, "beta0": beta0, "betas": betas}
            )

        # Select the best lambda
        scores = [result["score"] for result in results]
        scores = np.array(scores)
        idx = np.argmax(scores)

        self.lmbda = results[idx]["lmbda"]
        self.beta0 = results[idx]["beta0"]
        self.betas = results[idx]["betas"]

        self.fitted = True

        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts on the input X"""
        ...

    def fit_predict(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        test_X: np.ndarray,
    ) -> np.ndarray:
        """Fits the model on the train set and predicts on the test set"""
        ...
