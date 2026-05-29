"""Aqueous-solubility (LogS) scorer backed by an AutoGluon tabular regressor.

A regression twin of :class:`agfn.autogluon.bbb.AutoGluonBBBScorer`: it shares the MACCS +
RDKit-descriptor featurization but calls ``predict`` (continuous LogS) instead of
``predict_proba``. Higher LogS = more soluble.
"""

import numpy as np

from agfn.autogluon.featurization import generate_features


class AutoGluonSolubilityScorer:
    def __init__(self, model_path, predictor=None):
        self.model_path = model_path
        self._predictor = predictor
        if predictor is None:
            # Resolve eagerly so a missing dependency / unloadable model aborts at driver
            # startup instead of silently scoring every molecule NaN during training. Tests
            # inject a predictor and skip this path.
            _ = self.predictor

    @property
    def predictor(self):
        if self._predictor is None:
            try:
                from autogluon.tabular import TabularPredictor
            except ImportError as e:
                raise ImportError(
                    "Solubility constraint requires autogluon.tabular. Install the minimal "
                    "inference dependencies documented in docs/denovo_quickstart.md."
                ) from e
            # Saved under Python 3.13; the AGFN runtime is 3.12. Tree-based tabular ensembles
            # load safely across minor Python versions, so bypass the strict check.
            self._predictor = TabularPredictor.load(
                self.model_path, require_py_version_match=False)
        return self._predictor

    def score_smiles(self, smiles_list):
        if len(smiles_list) == 0:
            return np.array([], dtype=np.float64)
        features = generate_features(smiles_list)
        return np.asarray(self.predictor.predict(features), dtype=np.float64)

    def score_smiles_with_failures(self, smiles_list):
        """Return LogS aligned to ``smiles_list`` plus a per-row success mask.

        Rows whose features or predictor output cannot be computed keep a NaN score and
        ``False`` in the mask; the reward gate treats those as failures.
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

        # Exceptional path only: salvage good rows one at a time if a mixed batch errors.
        for row_index in feature_indices:
            row_score = self._predict_feature_rows(features.iloc[[row_index]])
            if row_score is None or len(row_score) != 1 or not np.isfinite(row_score[0]):
                continue
            scores[row_index] = row_score[0]
            score_success[row_index] = True
        return scores, score_success

    def _predict_feature_rows(self, features):
        try:
            preds = self.predictor.predict(features.reset_index(drop=True))
            scores = np.asarray(preds, dtype=np.float64)
            if len(scores) != len(features):
                return None
            return scores
        except Exception:
            return None
