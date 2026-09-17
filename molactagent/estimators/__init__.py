"""sklearn-style estimator wrappers around the MolActAgent engine.

These wrappers exist to make MolActAgent a first-class member of the
scikit-learn ecosystem — they implement ``fit`` / ``predict`` /
``predict_proba`` / ``score`` and follow the
``BaseEstimator`` + mixin contract, so they slot directly into
``cross_val_score``, ``GridSearchCV``, ``Pipeline``, and any other
sklearn machinery.

They are intentionally **thin**. A wrapper's only job is to:

1. Construct a :class:`molactagent.agent.MolActAgent` (or the
   regression backend it delegates to) with the right hyperparameters
   at ``fit`` time.
2. Forward ``X`` / ``y`` into the engine.
3. Project the engine's per-sample result dicts into the array-of-
   labels (or array-of-floats) shapes the sklearn contract expects.
4. Expose the rich evidence as :mod:`molactagent.result` wrappers
   when the caller asks for it.

No prediction logic, no aggregation, no confidence calibration lives
here. All of that stays in the engine; see ``docs/design.md`` Part
III. The wrappers are pure projections.

Why a separate module?

- The engine API (``MolActAgent.fit(X, y, task=...)`` returning rich
  dicts) is the right shape for the system's internals but the wrong
  shape for sklearn interop. A thin estimator layer fixes the
  mismatch without polluting the engine with sklearn imports.
- Wrappers are easy to add and remove; the engine is not. Keeping
  them physically separate keeps the engine's public surface small.
"""

from .classifier import MolActAgentClassifier
from .regressor import MolActAgentRegressor

__all__ = ["MolActAgentClassifier", "MolActAgentRegressor"]
