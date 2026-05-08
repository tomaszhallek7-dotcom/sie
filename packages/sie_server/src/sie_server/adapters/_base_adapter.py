from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING, Any, ClassVar, cast

from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED
from sie_server.adapters._utils import grouped_score_pairs
from sie_server.adapters.base import ModelAdapter, ModelCapabilities, ModelDims

if TYPE_CHECKING:
    import torch

    from sie_server.core.inference_output import ScoreOutput
    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)


class BaseAdapter(ModelAdapter):
    """Concrete base with common defaults.

    Provides:
    - ``capabilities`` / ``dims`` properties derived from ``spec``.
    - Standard ``unload()`` driven by ``spec.unload_fields``.
    - Default ``get_preprocessor()`` returning ``CharCountPreprocessor``.
    - ``_resolve_dtype()`` mapping ``compute_precision`` string to dtype.
    - ``_check_loaded()`` guard for encode/score/extract entry points.

    Every concrete subclass must declare a class-level ``spec``.
    """

    spec: ClassVar[AdapterSpec]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # Only validate classes that declare their own spec
        if "spec" not in cls.__dict__:
            return

        spec = cls.spec
        if not isinstance(spec, AdapterSpec):
            msg = f"{cls.__name__}.spec must be an AdapterSpec instance"
            raise TypeError(msg)

        if not spec.inputs:
            msg = f"{cls.__name__}.spec.inputs must be non-empty"
            raise TypeError(msg)

        if not spec.outputs:
            msg = f"{cls.__name__}.spec.outputs must be non-empty"
            raise TypeError(msg)

        # Validate output -> method consistency
        encode_outputs = {"dense", "sparse", "multivector"}
        declared_encode = encode_outputs & set(spec.outputs)
        if declared_encode and cls.encode is ModelAdapter.encode:
            msg = f"{cls.__name__} declares {declared_encode} in outputs but does not implement encode()"
            raise TypeError(msg)

        if "score" in spec.outputs:
            # BaseAdapter ships a default score_pairs() that delegates to score().
            # Treat that default as "not implemented" for validation purposes:
            # subclasses must override either score() or score_pairs() so the
            # default delegate doesn't bottom out in ModelAdapter.score().
            score_overridden = cls.score is not ModelAdapter.score
            score_pairs_overridden = cls.score_pairs not in (
                ModelAdapter.score_pairs,
                BaseAdapter.score_pairs,
            )
            if not score_overridden and not score_pairs_overridden:
                msg = f"{cls.__name__} declares 'score' in outputs but does not implement score() or score_pairs()"
                raise TypeError(msg)

        if "json" in spec.outputs and cls.extract is ModelAdapter.extract:
            msg = f"{cls.__name__} declares 'json' in outputs but does not implement extract()"
            raise TypeError(msg)

    # -- Properties derived from spec ----------------------------------------

    @property
    def capabilities(self) -> ModelCapabilities:
        # spec stores Literal tuples; cast needed because list() widens type.
        return ModelCapabilities(
            inputs=cast("Any", list(self.spec.inputs)),
            outputs=cast("Any", list(self.spec.outputs)),
        )

    @property
    def dims(self) -> ModelDims:
        return ModelDims(
            dense=self.spec.dense_dim or getattr(self, "_dense_dim", None),
            sparse=self.spec.sparse_dim or getattr(self, "_sparse_dim", None),
            multivector=self.spec.multivector_dim or getattr(self, "_multivector_dim", None),
        )

    # -- Standard lifecycle --------------------------------------------------

    def unload(self) -> None:
        """Unload model weights and free device memory.

        Iterates ``spec.unload_fields`` and sets each to ``None``, then
        runs ``gc.collect()`` and clears the device cache.
        """
        device = getattr(self, "_device", None)

        for attr in self.spec.unload_fields:
            if hasattr(self, attr):
                setattr(self, attr, None)

        self._device = None

        gc.collect()

        if device is not None:
            import torch as _torch

            if str(device).startswith("cuda"):
                _torch.cuda.empty_cache()
            elif str(device) == "mps":
                _torch.mps.empty_cache()

    def get_preprocessor(self) -> Any:
        """Return ``CharCountPreprocessor`` for cost estimation."""
        from sie_server.core.preprocessor import CharCountPreprocessor

        return CharCountPreprocessor(
            model_name=getattr(self, "_model_name_or_path", ""),
        )

    # -- Default batched scoring ---------------------------------------------

    def score_pairs(
        self,
        queries: list[Item],
        docs: list[Item],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> ScoreOutput:
        """Default ``score_pairs()`` that batches via per-query grouping.

        Groups parallel ``(query, doc)`` pairs by ``(text, id, instruction)``
        so each unique query is encoded once and its docs are scored as a
        single ``score()`` call. Subclasses with a more efficient native
        cross-batch path (e.g. cross-encoders that pack queries and docs
        into one transformer pass) should override this.

        Per-call ``options`` are not supported by this default delegate
        (it dispatches per-query and cannot route options into ``score()``
        without subclass-specific knowledge). If ``options`` is a non-empty
        mapping, this raises ``NotImplementedError`` to surface the
        unsupported configuration; pass ``options=None`` (or ``{}``) or
        override ``score_pairs()`` with an options-aware implementation.
        """
        if options:
            msg = (
                f"{type(self).__name__}.score_pairs(): per-call options are "
                f"not supported by the default batching path "
                f"(got options={options!r}). Override score_pairs() with an "
                f"options-aware implementation."
            )
            raise NotImplementedError(msg)
        return grouped_score_pairs(self.score, queries, docs, instruction=instruction)

    # -- Shared helpers ------------------------------------------------------

    def _check_loaded(self) -> None:
        """Raise ``RuntimeError`` if the model is not loaded."""
        if getattr(self, "_model", None) is None:
            raise RuntimeError(ERR_NOT_LOADED)

    def _resolve_dtype(self) -> torch.dtype:
        """Map ``self._compute_precision`` to a ``torch.dtype``."""
        import torch as _torch

        dtype_map: dict[str, torch.dtype] = {
            "float16": _torch.float16,
            "bfloat16": _torch.bfloat16,
            "float32": _torch.float32,
        }
        return dtype_map.get(
            getattr(self, "_compute_precision", "float16"),
            _torch.float16,
        )
