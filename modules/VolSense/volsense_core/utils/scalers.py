import torch
import numpy as np
import pandas as pd
from typing import Any, Optional, List


class TorchStandardScaler:
    """
    Minimal sklearn-like scaler implemented with PyTorch tensors.

    Records feature names when fit() is called with a pandas.DataFrame,
    returns numpy arrays from transform()/inverse_transform() (pandas-friendly),
    and exposes state_dict()/load_state_dict() for checkpoint persistence.

    :ivar mean_: Row vector (1, n_features) of feature means as torch.Tensor.
    :ivar std_: Row vector (1, n_features) of feature std devs as torch.Tensor.
    :ivar feature_names_in_: Optional ordered list of feature names seen during fit().
    """

    def __init__(self) -> None:
        """Initialize empty scaler state."""
        self.mean_: Optional[torch.Tensor] = None
        self.std_: Optional[torch.Tensor] = None
        self.feature_names_in_: Optional[List[str]] = None

    def _is_dataframe(self, X: Any) -> bool:
        """
        Determine whether X is a pandas DataFrame.

        :param X: Input to check.
        :return: True if X is a pandas.DataFrame, else False.
        """
        return isinstance(X, pd.DataFrame)

    def _to_tensor(self, X: Any) -> torch.Tensor:
        """
        Convert input to a torch.float32 tensor.

        If X is a DataFrame, record column names in feature_names_in_ and
        convert the underlying numpy values to a torch tensor.

        :param X: Array-like / DataFrame / torch.Tensor input
        :return: torch.Tensor of dtype float32
        """
        if self._is_dataframe(X):
            # preserve input column ordering
            self.feature_names_in_ = list(X.columns)
            arr = X.values.astype(float)
            return torch.as_tensor(arr, dtype=torch.float32)
        if isinstance(X, np.ndarray):
            return torch.as_tensor(X.astype(float), dtype=torch.float32)
        if isinstance(X, torch.Tensor):
            return X.to(torch.float32)
        # fallback conversion
        return torch.as_tensor(np.asarray(X, dtype=float), dtype=torch.float32)

    def fit(self, X: Any) -> "TorchStandardScaler":
        """
        Compute the per-feature mean and std dev from X and store them.

        If X is a pandas.DataFrame, the column names are recorded in
        feature_names_in_ for later alignment.

        :param X: Training data (DataFrame, ndarray or tensor) with shape (n_samples, n_features)
        :return: self
        """
        Xt = self._to_tensor(X)
        # store as row vectors (1, n_features)
        self.mean_ = Xt.mean(dim=0, keepdim=True)
        # unbiased=False to match sklearn's default population std behaviour for transformers
        self.std_ = Xt.std(dim=0, keepdim=True, unbiased=False)
        return self

    def transform(self, X: Any) -> np.ndarray:
        """
        Scale X using stored mean_ and std_.

        :param X: Data to transform (DataFrame/ndarray/tensor). Must have same number of features as fitted data.
        :raises RuntimeError: if scaler has not been fitted.
        :return: numpy.ndarray of transformed data (shape: n_samples, n_features)
        """
        Xt = self._to_tensor(X)
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler must be fitted before calling transform()")
        out = (Xt - self.mean_) / (self.std_ + 1e-8)
        return out.cpu().numpy()

    def fit_transform(self, X: Any) -> np.ndarray:
        """
        Fit to X then transform X.

        :param X: Data to fit and transform
        :return: numpy.ndarray of transformed data
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: Any) -> np.ndarray:
        """
        Undo the scaling operation, returning data on the original scale.

        :param X: Scaled data (array-like or tensor)
        :raises RuntimeError: if scaler has not been fitted.
        :return: numpy.ndarray of data on original scale
        """
        Xt = self._to_tensor(X)
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError(
                "Scaler must be fitted before calling inverse_transform()"
            )
        out = Xt * (self.std_ + 1e-8) + self.mean_
        return out.cpu().numpy()

    def state_dict(self) -> dict:
        """
        Return scaler state as a JSON-serializable dictionary.

        This method enables checkpoint persistence by exporting the scaler's
        internal parameters (mean, std, feature names) in a format that can
        be saved to JSON and later restored via load_state_dict().

        :return: Dictionary containing 'mean_', 'std_', and 'feature_names_in_'.
                 Tensor values are converted to lists for JSON compatibility.
        :rtype: dict
        """
        return {
            "mean_": self.mean_.tolist() if self.mean_ is not None else None,
            "std_": self.std_.tolist() if self.std_ is not None else None,
            "feature_names_in_": self.feature_names_in_
        }

    def load_state_dict(self, state: dict) -> None:
        """
        Restore scaler state from a dictionary.

        This method reconstructs the scaler's internal parameters from a
        state dictionary previously created by state_dict(). Used during
        model checkpoint loading.

        :param state: Dictionary containing 'mean_', 'std_', and optionally 'feature_names_in_'.
        :type state: dict
        :return: None
        :rtype: None
        """
        if state.get("mean_") is not None:
            self.mean_ = torch.tensor(state["mean_"], dtype=torch.float32)
        if state.get("std_") is not None:
            self.std_ = torch.tensor(state["std_"], dtype=torch.float32)
        if "feature_names_in_" in state:
            self.feature_names_in_ = state["feature_names_in_"]
