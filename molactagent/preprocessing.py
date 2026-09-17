"""Internal feature scaling and missing-value imputation.

This module is the LEGACY internal preprocessor. It exposes
:class:`DataPreprocessor`, a thin wrapper around scikit-learn's
``StandardScaler`` / ``MinMaxScaler`` plus column-median imputation
for NaNs. It is consumed by :mod:`molactagent.agent.MolActAgent.fit`
to produce the ``X_ref_scaled`` matrix that downstream contextual-
similarity computations operate on.

Naming caveat for users of the v0.3 layered API:

This module name (``molactagent.preprocessing``) collides with the
Scanpy-style namespace path that would otherwise expose
:mod:`molactagent.pp` as ``molactagent.preprocessing``. Because Python
resolves submodule attributes before falling back to ``__getattr__``,
``molactagent.preprocessing`` resolves to **this file** rather than
to ``molactagent.pp``. The user-facing preprocessing namespace is
therefore exposed only as ``molactagent.pp`` (no full-word alias).

This is a deliberate, documented choice: renaming this internal
module would break ``molactagent.agent`` and any external code that
imported ``DataPreprocessor`` directly. See ``docs/design.md`` §11.2
for the layered-API discussion and the rationale.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, scaler_type: str = "standard"):
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif scaler_type == "none":
            self.scaler = None
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        self.scaler_type = scaler_type
        self._medians = None
        self._missing_features: set[int] = set()

    def fit_transform(self, X_ref: np.ndarray) -> np.ndarray:
        X = np.array(X_ref, dtype=np.float64)

        missing_mask = np.isnan(X)
        if missing_mask.any():
            self._medians = np.nanmedian(X, axis=0)
            missing_cols = np.where(missing_mask.any(axis=0))[0]
            self._missing_features = set(missing_cols.tolist())
            for col in missing_cols:
                X[missing_mask[:, col], col] = self._medians[col]
            logger.info(
                f"Filled missing values in {len(missing_cols)} features using median"
            )
        else:
            self._medians = np.median(X, axis=0)

        if self.scaler is not None:
            return self.scaler.fit_transform(X)
        return X.copy()

    def transform(self, X_test: np.ndarray) -> np.ndarray:
        X = np.array(X_test, dtype=np.float64)

        missing_mask = np.isnan(X)
        if missing_mask.any() and self._medians is not None:
            for col in range(X.shape[1]):
                col_missing = missing_mask[:, col]
                if col_missing.any():
                    X[col_missing, col] = self._medians[col]

        if self.scaler is not None:
            return self.scaler.transform(X)
        return X.copy()
