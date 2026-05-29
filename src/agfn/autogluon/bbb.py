"""Blood-brain-barrier (BBB) permeability scorer backed by an AutoGluon tabular model."""

import numpy as np
import pandas as pd

from agfn.autogluon.featurization import generate_features


class AutoGluonBBBScorer:
    def __init__(self, model_path, positive_class=1, predictor=None):
        self.model_path = model_path
        self.positive_class = positive_class
        self._predictor = predictor
        if predictor is None:
            # Resolve the predictor eagerly so a missing dependency or an unloadable model
            # aborts the run at construction time (driver startup) instead of silently
            # degrading every molecule to a NaN BBB score during training. Tests inject a
            # predictor and skip this path.
            _ = self.predictor

    @property
    def predictor(self):
        if self._predictor is None:
            try:
                from autogluon.tabular import TabularPredictor
            except ImportError as e:
                raise ImportError(
                    "BBB constraint requires autogluon.tabular. Install the minimal inference "
                    "dependencies documented in docs/denovo_quickstart.md."
                ) from e
            # The saved models were trained under Python 3.13; the AGFN runtime env is
            # Python 3.12. Tree-based tabular ensembles (CatBoost/LightGBM/XGBoost) load
            # safely across minor Python versions, so bypass AutoGluon's strict check.
            self._predictor = TabularPredictor.load(
                self.model_path, require_py_version_match=False)
        return self._predictor

    def score_smiles(self, smiles_list):
        if len(smiles_list) == 0:
            return np.array([], dtype=np.float64)
        features = generate_features(smiles_list)
        probabilities = self.predictor.predict_proba(features)
        return self._positive_probability(probabilities)

    def score_smiles_with_failures(self, smiles_list):
        """Return BBB scores aligned to ``smiles_list`` plus a per-row success mask.

        Rows whose BBB features or predictor output cannot be computed keep a NaN score and
        ``False`` in the success mask. The reward code treats those rows as BBB failures.
        """
        n = len(smiles_list)
        scores = np.full(n, np.nan, dtype=np.float64)
        score_success = np.zeros(n, dtype=bool)
        if n == 0:
            return scores, score_success

        features, feature_success = generate_features(smiles_list, return_success=True)
        feature_indices = np.flatnonzero(feature_success)
        if len(feature_indices) == 0:
            return scores, score_success

        scored = self._predict_feature_rows(features.iloc[feature_indices])
        if scored is not None:
            finite = np.isfinite(scored)
            scores[feature_indices] = scored
            score_success[feature_indices] = finite
            return scores, score_success

        # Exceptional path only: if AutoGluon rejects a mixed batch, salvage good rows one at a
        # time while failing closed for rows that still error.
        for row_index in feature_indices:
            row_score = self._predict_feature_rows(features.iloc[[row_index]])
            if row_score is None or len(row_score) != 1 or not np.isfinite(row_score[0]):
                continue
            scores[row_index] = row_score[0]
            score_success[row_index] = True
        return scores, score_success

    def _predict_feature_rows(self, features):
        try:
            probabilities = self.predictor.predict_proba(features.reset_index(drop=True))
            scores = self._positive_probability(probabilities)
            if len(scores) != len(features):
                return None
            return scores
        except Exception:
            return None

    def _positive_probability(self, probabilities):
        if not isinstance(probabilities, pd.DataFrame):
            probabilities = pd.DataFrame(probabilities)
        for column in self._positive_class_candidates():
            if column in probabilities.columns:
                return probabilities[column].to_numpy(dtype=np.float64)
        raise KeyError(
            f"Positive class {self.positive_class!r} not found in AutoGluon probability "
            f"columns {list(probabilities.columns)!r}."
        )

    def _positive_class_candidates(self):
        candidates = [self.positive_class, str(self.positive_class)]
        try:
            candidates.append(int(self.positive_class))
        except (TypeError, ValueError):
            pass
        deduped = []
        for candidate in candidates:
            if candidate not in deduped:
                deduped.append(candidate)
        return deduped
