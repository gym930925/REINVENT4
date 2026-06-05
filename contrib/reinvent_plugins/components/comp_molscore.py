"""MolScore adapter for REINVENT scoring

Wraps any scoring function exposed by MolScore for use as a REINVENT
scoring component.  The user specifies the MolScore scoring function
class name and its keyword arguments via the REINVENT config.
"""

from __future__ import annotations

__all__ = ["MolScore"]

import logging
import tempfile
from typing import List, Optional, Dict, Any

import numpy as np
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from .add_tag import add_tag
from reinvent_plugins.normalize import normalize_smiles

logger = logging.getLogger("reinvent")


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the MolScore adapter component

    Note that all parameters are always lists because components can have
    multiple endpoints and so all the parameters from each endpoint is
    collected into a list.  This is also true in cases where there is only one
    endpoint.
    """

    # The MolScore scoring function class name, e.g. "MolecularDescriptors"
    scoring_function: List[str]
    # The metric to extract from the scoring function results, e.g. "QED"
    metric: List[str]
    # Prefix for the MolScore scoring function (used for namespacing metrics)
    prefix: List[str]
    # JSON-encoded string of extra kwargs to pass to the MolScore scoring function
    kwargs: Optional[List[str]] = None


@add_tag("__component")
class MolScore:
    """Adapter to use any MolScore scoring function in REINVENT

    MolScore (https://github.com/MorganCThomas/MolScore) exposes many
    scoring functions (descriptors, similarity, docking, QSAR, etc.).
    This component wraps any of them by:

    1. Instantiating the requested MolScore scoring function class
    2. Calling it with the batch of SMILES
    3. Extracting the requested metric from the results

    Example TOML config:
        [[scoring.component]]
        [scoring.component.MolScore]

        [[scoring.component.MolScore.endpoint]]
        name = "QED"
        weight = 1.0

        [scoring.component.MolScore.endpoint.params]
        scoring_function = "MolecularDescriptors"
        prefix = "desc"
        metric = "QED"
        kwargs = '{"n_jobs": 1}'

        [scoring.component.MolScore.endpoint.transform]
        type = "sigmoid"
        high = 0.9
        low = 0.3
        k = 0.5
    """

    def __init__(self, params: Parameters):
        import json

        from molscore.scoring_functions import all_scoring_functions

        self.smiles_type = "rdkit_smiles"
        self.scorers = []
        self.metrics = []

        for i, sf_name in enumerate(params.scoring_function):
            prefix = params.prefix[i]
            metric = params.metric[i]

            # Parse extra kwargs
            extra_kwargs: Dict[str, Any] = {}
            if params.kwargs and i < len(params.kwargs) and params.kwargs[i]:
                extra_kwargs = json.loads(params.kwargs[i])

            # Look up the scoring function class in MolScore's registry
            scorer_cls = None
            for cls in all_scoring_functions:
                if cls.__name__ == sf_name:
                    scorer_cls = cls
                    break

            if scorer_cls is None:
                available = [cls.__name__ for cls in all_scoring_functions]
                raise ValueError(
                    f"MolScore scoring function '{sf_name}' not found. "
                    f"Available: {available}"
                )

            # Instantiate the scoring function
            scorer = scorer_cls(prefix=prefix, **extra_kwargs)
            self.scorers.append(scorer)
            # Build the full metric key as MolScore uses "{prefix}_{metric}"
            self.metrics.append(f"{prefix}_{metric}")

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> ComponentResults:
        all_scores = []

        for scorer, metric_key in zip(self.scorers, self.metrics):
            # MolScore scoring functions accept smiles and return List[Dict]
            # They also accept directory and file_names but those are optional
            # for non-file-based scorers
            with tempfile.TemporaryDirectory() as tmpdir:
                results = scorer(
                    smiles=smilies,
                    directory=tmpdir,
                    file_names=[f"mol_{i}" for i in range(len(smilies))],
                )

            # Extract the requested metric from results
            scores = []
            for result_dict in results:
                value = result_dict.get(metric_key, np.nan)
                try:
                    scores.append(float(value))
                except (TypeError, ValueError):
                    scores.append(np.nan)

            all_scores.append(np.array(scores, dtype=np.float64))

        return ComponentResults(all_scores)
