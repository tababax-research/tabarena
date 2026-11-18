from __future__ import annotations

import logging
import tempfile
from collections import OrderedDict
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from tabarena.models.tababax.hw_plan import (
    begin_strategy,
    member_row_index,
    plan_context,
    plan_query,
    read_hardware,
    update_actual,
)

from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.core.models.abstract import SharedWeights
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def _keep_top_variance_columns(X: "pd.DataFrame", k: int) -> "pd.DataFrame":
    """Keep the ``k`` columns with the largest variance. Ties break on the column name."""
    if k >= X.shape[1]:
        return X
    scores: list[tuple[float, str]] = []
    for col in X.columns:
        series = X[col]
        if not np.issubdtype(series.dtype, np.number):
            scores.append((0.0, str(col)))
            continue
        variance = float(np.nanvar(series.to_numpy(dtype=np.float64, copy=False)))
        if not np.isfinite(variance):
            variance = 0.0
        scores.append((variance, str(col)))
    order = sorted(range(len(scores)), key=lambda i: (-scores[i][0], scores[i][1]))
    keep = [X.columns[i] for i in order[:k]]
    return X.loc[:, keep]


def _transform_member_rows(generator, X, mode: str, index: np.ndarray):
    """EnsembleGenerator.transform, but each member sees its own row slice of the context.

    Test rows are not sliced here. Query chunking, when it is on, already cut them.
    """
    from sklearn.utils.validation import check_is_fitted

    check_is_fitted(generator, ["ensemble_configs_"])
    if mode not in ("both", "train", "test"):
        raise ValueError(f"Invalid mode: {mode}")

    def _member_rows(member: int) -> np.ndarray:
        return index[member % index.shape[0]]

    if mode == "train":
        y = generator.y_
        data = OrderedDict()
        member = 0
        for norm_method, shuffle_configs in generator.ensemble_configs_.items():
            X_preprocessed = generator.preprocessors_[norm_method].X_transformed_
            X_ensemble = []
            y_ensemble = []
            for feat_shuffle, y_pattern in shuffle_configs:
                rows = _member_rows(member)
                member += 1
                X_ensemble.append(X_preprocessed[rows][:, feat_shuffle])
                y_member = y[rows]
                if generator.classification:
                    y_ensemble.append(np.array(y_pattern)[y_member.astype(int)])
                else:
                    y_ensemble.append(y_member)
            data[norm_method] = (np.stack(X_ensemble, axis=0), np.stack(y_ensemble, axis=0))
        return data

    if X is None:
        raise ValueError("X is required when mode is 'test' or 'both'")
    X = generator.unique_filter_.transform(X)

    if mode == "test":
        data = OrderedDict()
        for norm_method, shuffle_configs in generator.ensemble_configs_.items():
            X_test_preprocessed = generator.preprocessors_[norm_method].transform(X)
            X_ensemble = [X_test_preprocessed[:, feat_shuffle] for feat_shuffle, _ in shuffle_configs]
            data[norm_method] = (np.stack(X_ensemble, axis=0),)
        return data

    y = generator.y_
    data = OrderedDict()
    member = 0
    for norm_method, shuffle_configs in generator.ensemble_configs_.items():
        preprocessor = generator.preprocessors_[norm_method]
        X_train_pp = preprocessor.X_transformed_
        X_test_pp = preprocessor.transform(X)
        X_ensemble = []
        y_ensemble = []
        for feat_shuffle, y_pattern in shuffle_configs:
            rows = _member_rows(member)
            member += 1
            X_variant = np.concatenate([X_train_pp[rows], X_test_pp], axis=0)
            X_ensemble.append(X_variant[:, feat_shuffle])
            y_member = y[rows]
            if generator.classification:
                y_ensemble.append(np.array(y_pattern)[y_member.astype(int)])
            else:
                y_ensemble.append(y_member)
        data[norm_method] = (np.stack(X_ensemble, axis=0), np.stack(y_ensemble, axis=0))
    return data


def _install_member_row_patch() -> None:
    """Teach TabAbax's ensemble to read ``_hw_row_index`` set by the wrapper."""
    from tababax._sklearn.preprocessing import EnsembleGenerator

    if getattr(EnsembleGenerator, "_hw_member_rows", False):
        return
    orig_fit = EnsembleGenerator.fit
    orig_transform = EnsembleGenerator.transform

    def fit(self, X, y, __orig=orig_fit):
        pending = getattr(EnsembleGenerator, "_hw_pending_row_index", None)
        self._hw_row_index = None if pending is None else np.asarray(pending, dtype=np.int64)
        return __orig(self, X, y)

    def transform(self, X=None, mode="both", __orig=orig_transform):
        index = getattr(self, "_hw_row_index", None)
        if index is None:
            return __orig(self, X, mode)
        return _transform_member_rows(self, X, mode, index)

    EnsembleGenerator.fit = fit
    EnsembleGenerator.transform = transform
    EnsembleGenerator._hw_member_rows = True
    EnsembleGenerator._hw_pending_row_index = None

_HF_REPO_ID = "tababax-ai/TabAbax"
#: Released checkpoint per task type; the library selects the same two files by default.
_HF_CHECKPOINTS = {
    "classification": "tababax-classifier.ckpt",
    "regression": "tababax-regressor.ckpt",
}
#: Commit that uploaded the released weights. Later commits only touched the model card, but the
#: library resolves ``main``, so pinning here keeps a republished checkpoint from changing the run.
#: Public so a run script can pin the same commit without importing a private name.
HF_REPO_ID = _HF_REPO_ID
HF_CHECKPOINTS = _HF_CHECKPOINTS
CHECKPOINT_REVISION = "039e4f239d06a4d631e38c15bc23ef195ed8cf79"


class TabAbaxModel(AbstractTorchModel):
    """TabAbax, an in-context foundation model for classification and regression.

    ``fit`` stores the training table. Each ensemble member runs one forward pass.
    The default estimator uses eight dataset-view ensemble members. The weights
    are not updated on the task.

    Omitting ``model_path`` loads the released checkpoint from
    ``tababax-ai/TabAbax`` at the commit this wrapper pins.
    Classification uses ``tababax-classifier.ckpt``
    and regression uses ``tababax-regressor.ckpt``. Pass ``model_path`` to load a
    local file instead. The weights are CC-BY-NC-4.0 and are limited to non-commercial use.

    Recommended defaults, applied unless a hyperparameter overrides them:

    - ``n_estimators=8``: ensemble views. ``1`` is a single forward pass.
    - ``batch_size=1``: ensemble members forwarded together. ``None`` sends all of them at once.
    - ``offload_mode="auto"``: puts the large column embedding on GPU, CPU, or disk according to free memory.
    - ``use_amp="auto"``: mixed precision from the table size. Off for small tables and on CPU.
    - ``use_fa3=False``: FlashAttention-3. Leave off unless the runtime provides it.
    - ``allow_auto_download=True``: download the checkpoint when ``model_path`` is missing.
      ``False`` refuses the download.

    ``random_state`` is the TabArena seed (``seed_name``). ``device`` follows the GPUs AutoGluon
    allocated: CUDA when ``num_gpus`` is not 0, otherwise CPU.

    ``time_limit`` is accepted by the AutoGluon interface for API compatibility,
    but it does not bound fitting for this in-context model. Fitting performs
    inference with a released checkpoint; there is no task-level training loop,
    validation split, early stopping, or hyperparameter search to interrupt.

    The TabArena integration imports TabAbax, whose inference implementation
    contains portions derived from TabICLv2 and distributed under BSD-3-Clause.
    The additional TabAbax packaging and integration code is Apache-2.0.
    The default checkpoints are separate CC-BY-NC-4.0 artifacts and are not
    commercially licensed.

    Authors: TabAbax, based on TabICL by Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan
    Codebase: https://github.com/tababax-research/TabAbax
    Upstream: https://github.com/soda-inria/tabicl
    Weights: https://huggingface.co/tababax-ai/TabAbax
    License: BSD-3-Clause for the TabICL-derived code. Apache-2.0 for TabAbax packaging. Weights: CC-BY-NC-4.0.
    """

    ag_key = "TA-TABABAX"
    ag_name = "TA-TabAbax"
    ag_priority = 65
    seed_name = "random_state"
    warmup_modules: ClassVar[tuple[str, ...]] = ("tababax",)
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_resources_physical_cores_only = True
    default_num_gpus = 1
    minimum_num_gpus = 1
    _default_ag_args_ensemble_extra = {"fold_fitting_strategy": "sequential_local", "refit_folds": True}
    #: The estimator builds its network inside ``_load_model``, which ``fit`` calls. The first
    #: fit of a checkpoint in the process builds it; every later estimator of that checkpoint
    #: and device reuses it.
    #: ``checkpoint_revision`` is part of the key: the same file name at two Hub commits is two
    #: different networks, and sharing one would score a second revision on the first one's weights.
    shared_weights: ClassVar[SharedWeights] = SharedWeights(
        loader=(
            "tababax:TabAbaxClassifier._load_model",
            "tababax:TabAbaxRegressor._load_model",
        ),
        key=("checkpoint_version", "checkpoint_revision", "model_path"),
    )
    #: Knobs that make the warm-up's dummy fit cheap without touching the network.
    cheap_hyperparameters: ClassVar[dict] = {"n_estimators": 1}

    def get_model_cls(self):
        if self.problem_type in ["binary", "multiclass"]:
            from tababax import TabAbaxClassifier

            return TabAbaxClassifier
        from tababax import TabAbaxRegressor

        return TabAbaxRegressor

    @classmethod
    def download_checkpoint(cls, task: str) -> str:
        """Resolve one released checkpoint (``"classification"`` / ``"regression"``) at the pinned revision.

        The Hub cache answers first (no network round trip on a node that ran ``prefetch_weights``),
        a download otherwise.
        """
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        kwargs = {"repo_id": _HF_REPO_ID, "filename": _HF_CHECKPOINTS[task], "revision": CHECKPOINT_REVISION}
        try:
            return hf_hub_download(**kwargs, local_files_only=True)
        except LocalEntryNotFoundError:
            return hf_hub_download(**kwargs)

    @classmethod
    def prefetch_weights(cls) -> dict[str, str]:
        """Pre-download both released checkpoints; return ``{task: local path}``.

        Classification and regression are separate files, so a run covering both problem types
        needs both warmed before the jobs are dispatched.
        """
        return {task: cls.download_checkpoint(task) for task in _HF_CHECKPOINTS}

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        time_limit: float | None = None,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        # Context fitting runs one forward pass per ensemble member: there is no iterative
        # training, early stopping or validation split to cut short, so ``time_limit`` cannot
        # truncate the fit without dropping ensemble members. It is accepted and left unused.
        try:
            import tababax  # noqa: F401
        except ImportError as err:
            logger.log(
                40,
                "\tFailed to import tababax! Install it with `pip install tabarena[tababax]`. "
                "The package is derived from TabICL (github.com/soda-inria/tabicl, BSD-3-Clause).",
            )
            raise err

        from torch.cuda import is_available

        device = "cuda" if num_gpus != 0 else "cpu"
        if (device == "cuda") and (not is_available()):
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        hyp = self._get_model_params()
        task = "classification" if self.problem_type in ["binary", "multiclass"] else "regression"
        if "checkpoint_version" not in hyp:
            hyp["checkpoint_version"] = _HF_CHECKPOINTS[task]
        # Pin the revision the library resolves the checkpoint against. Without it the library follows
        # the Hub repo's default branch, so the bytes behind a published score could change under a
        # rerun. An explicit hyperparameter still wins, for a caller loading its own checkpoint.
        if hyp.get("checkpoint_revision") is None:
            hyp["checkpoint_revision"] = CHECKPOINT_REVISION
        disk_offload_dir = tempfile.mkdtemp(prefix="tababax_")
        self.model = self.get_model_cls()(
            **hyp,
            device=device,
            n_jobs=num_cpus,
            disk_offload_dir=disk_offload_dir,
            verbose=X.shape[0] > 250_000,
            inference_config=dict(
                # Row1/Col1 batches are independent by row/column; reusing their
                # shape-compatible input storage removes the second full
                # (T×tokens×128) host buffer without changing model arithmetic.
                # ColEmbedding and Row2 have different output shapes and fall back
                # to ordinary allocation automatically.
                # max_pinned_memory_mb=0: disable pinned-memory for large tensors.
                # With the default 32 GB pinned budget, the host-pinned region for
                # each member/stage is never freed during the forward pass, causing
                # RAM to exceed the machine limit on 100k+ row × 300+ col tables
                # (TabAbax issue #2).  Non-pinned allocation peaks at ~5× output
                # size but falls back to swap rather than OOM-killing the process.
                COL_CONFIG=dict(
                    cpu_safety_factor=0.75,
                    reuse_input_buffer=True,
                    use_async=False,
                    max_pinned_memory_mb=0,
                ),
                ROW_CONFIG=dict(
                    reuse_input_buffer=True,
                    use_async=False,
                    max_pinned_memory_mb=0,
                ),
                # offload=False: _build_inference_config (base.py) copies
                # offload_mode="auto" (a truthy string) into ICL_CONFIG.offload,
                # causing _inference_forward to move row representations ri to CPU
                # via `if mgr_config.offload: ri.cpu()`.  For >10-class tasks
                # _predict_hierarchical then receives CPU ri while group_probs is
                # on CUDA, raising a device mismatch.  Forcing offload=False keeps
                # ri on GPU.  Official TabAbax issue #2 workaround.
                # device=device: guards against .to(None) silently mapping to CPU
                # on some PyTorch builds when mgr_config.device is None.
                ICL_CONFIG=dict(offload=False, device=device),
            ),
        )
        X = self.preprocess(X, y=y)
        self._apply_hardware_plan(X, y, hyp)

    def _apply_hardware_plan(self, X: "pd.DataFrame", y: "pd.Series", hyp: dict) -> None:
        """Row-sample first, then column-sample, against the published GPU / RAM / disk budget.

        Row slices follow Kumo: member ``i`` gets a different permutation slice, so the
        ensemble's union covers more of the table than any single member. The forward
        still sees one member at a time (``batch_size=1``), so the peak buffer is one
        slice, not ``n_estimators`` slices. Columns are dropped only when one full-width
        row already exceeds the train budget. Query chunking is decided at predict time
        and stays off when context plus the whole test set fit in the same buffer.
        """
        from tababax._sklearn.preprocessing import EnsembleGenerator

        budget = read_hardware()
        n_estimators = int(hyp.get("n_estimators") or 8)
        seed = int(self.params.get("random_state", 0) or 0)
        plan = plan_context(X.shape[0], X.shape[1], n_estimators, budget)
        n_before_cols = int(X.shape[1])
        if plan["col_sample"]:
            X = _keep_top_variance_columns(X, plan["kept_cols"])
        row_index = None
        if plan["row_sample"]:
            row_index = member_row_index(X.shape[0], n_estimators, plan["kept_rows"], seed)
        coverage = 1.0
        if row_index is not None:
            coverage = float(np.unique(row_index).size) / float(X.shape[0])

        self._hw_budget = budget
        self._hw_n_cols = int(X.shape[1])
        self._hw_n_context = int(plan["kept_rows"] if plan["row_sample"] else X.shape[0])
        predicted = {
            "n_rows": int(X.shape[0]),
            "n_cols": n_before_cols,
            "row_sample": bool(plan["row_sample"]),
            "rows_per_member": int(self._hw_n_context),
            "col_sample": bool(plan["col_sample"]),
            "cols_kept": int(plan["kept_cols"]),
            "union_coverage": round(coverage, 4),
            "query_chunk": "at_predict",
        }
        begin_strategy(budget, predicted)
        update_actual(
            row_sample=predicted["row_sample"],
            rows_per_member=predicted["rows_per_member"],
            col_sample=bool(plan["col_sample"] and X.shape[1] < n_before_cols),
            cols_kept=int(X.shape[1]),
            union_coverage=round(coverage, 4),
            query_chunk=None,
            chunk_rows=None,
            n_test=None,
        )
        logger.log(
            20,
            "TabAbax hardware: gpu=%.1fGiB ram=%.1fGiB disk=%.1fGiB capacity=%.1fGiB "
            "train_budget=%.1fGiB | rows %d→%d/member cols %d→%d | union coverage %.1f%%",
            budget.gpu_bytes / 1024**3,
            budget.ram_bytes / 1024**3,
            budget.disk_bytes / 1024**3,
            budget.capacity_bytes / 1024**3,
            budget.train_budget_bytes / 1024**3,
            X.shape[0],
            self._hw_n_context,
            n_before_cols,
            X.shape[1],
            100.0 * coverage,
        )
        _install_member_row_patch()
        EnsembleGenerator._hw_pending_row_index = row_index
        try:
            self.model = self.model.fit(X=X, y=y)
        finally:
            EnsembleGenerator._hw_pending_row_index = None

    # Query chunking stays off unless context + the full test set exceeds capacity.
    # When it is on, each chunk is sized from the leftover bytes in that one buffer.
    # embed_with_test=False, so chunking does not change the prediction.
    #
    # Feature-name alignment: AutoGluon's preprocessing pipeline can produce columns
    # in a different ORDER between fit and predict for tables with text features
    # (NoCatAsStringCategoryFeatureGenerator expands to variable column orderings).
    # TabAbaxClassifier/Regressor.validate_data(reset=False) checks both name AND
    # order, raising ValueError if they differ.  We reorder X to the fitted order
    # before every predict call to prevent this.

    @staticmethod
    def _align_columns(X: "pd.DataFrame", model) -> "pd.DataFrame":
        """Reorder X columns to match the order the model was fitted on."""
        fitted_names = getattr(model, "feature_names_in_", None)
        if fitted_names is None:
            return X
        try:
            cols = list(fitted_names)
            if list(X.columns) != cols:
                return X[cols]
        except (KeyError, TypeError):
            pass  # let the model report the error with full details
        return X

    def _query_chunk_rows(self, n_test: int) -> int | None:
        """``None`` when the whole test set fits beside the context. Otherwise the chunk length."""
        budget = getattr(self, "_hw_budget", None)
        if budget is None:
            budget = read_hardware()
        n_context = int(getattr(self, "_hw_n_context", 0) or 0)
        n_cols = int(getattr(self, "_hw_n_cols", 0) or 0)
        query = plan_query(n_context, n_cols, n_test, budget)
        update_actual(
            query_chunk=bool(query["chunk"]),
            chunk_rows=int(query["chunk_rows"]),
            n_test=int(n_test),
            peak_gib=round(query["peak_bytes"] / 1024**3, 2),
        )
        if not query["chunk"]:
            return None
        return int(query["chunk_rows"])

    def _predict_proba(self, X: "pd.DataFrame", **kwargs):
        # Regression has no class probabilities. Some AG code paths call _predict_proba
        # for regression too; delegate so we never touch TabAbaxRegressor.predict_proba.
        if self.problem_type == "regression":
            return self._predict(X, **kwargs)
        X = self._align_columns(X, self.model)
        chunk = self._query_chunk_rows(X.shape[0])
        if chunk is None or X.shape[0] <= chunk:
            proba = self.model.predict_proba(X)
        else:
            proba = np.concatenate(
                [
                    self.model.predict_proba(X.iloc[i : i + chunk])
                    for i in range(0, X.shape[0], chunk)
                ],
                axis=0,
            )
        # Binary: AG's bagging writes each fold into a preallocated (N,) array.
        # TabAbaxClassifier.predict_proba returns (N, 2); the positive-class column is (N,).
        if self.problem_type == "binary":
            return proba[:, 1]
        return proba

    def _predict(self, X: "pd.DataFrame", **kwargs):
        X = self._align_columns(X, self.model)
        chunk = self._query_chunk_rows(X.shape[0])
        if chunk is None or X.shape[0] <= chunk:
            return self.model.predict(X)
        return np.concatenate(
            [self.model.predict(X.iloc[i : i + chunk]) for i in range(0, X.shape[0], chunk)],
            axis=0,
        )
    # --------------------------------------------------------------------------

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,
        **kwargs,
    ) -> int:
        """Host-memory estimate until the checkpoint's GPU offload behaviour is measured."""
        del hyperparameters, kwargs
        dataset_size_mem_est = 3 * get_approximate_df_mem_usage(X).sum()
        baseline_overhead_mem_est = 1e9
        return dataset_size_mem_est + baseline_overhead_mem_est

    def _set_default_params(self):
        """Defaults for a released checkpoint. A hyperparameter with the same name overrides these."""
        self._set_default_param_value("n_estimators", 8)
        self._set_default_param_value("batch_size", 1)
        self._set_default_param_value("offload_mode", "auto")
        self._set_default_param_value("use_amp", "auto")
        self._set_default_param_value("use_fa3", False)
        self._set_default_param_value("allow_auto_download", True)

    def get_device(self) -> str:
        return self.model.device_.type

    def _set_device(self, device: str):
        device = self.to_torch_device(device)
        self.model.device_ = device
        self.model.device = self.model.device_.type
        self.model.model_ = self.model.model_.to(self.model.device_)

        self.model.inference_config_.COL_CONFIG.device = self.model.device_
        self.model.inference_config_.ROW_CONFIG.device = self.model.device_
        self.model.inference_config_.ICL_CONFIG.device = self.model.device_

        # Recompute device-dependent settings such as use_amp="auto".
        self.model._build_inference_config()

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
