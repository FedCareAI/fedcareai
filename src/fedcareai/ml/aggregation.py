# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

import numpy as np
from numbers import Integral


class AggregationError(ValueError):
    pass


def weighted_average_weights(
    site_weights: list[list[np.ndarray]],
    sample_counts: list[int],
) -> list[np.ndarray]:
    """
    Compute weighted average of model weights.

    site_weights:
        [
          [layer1_weights_site1, layer2_weights_site1, ...],
          [layer1_weights_site2, layer2_weights_site2, ...]
        ]

    sample_counts:
        Number of local samples used by each site.
    """
    if not site_weights:
        raise AggregationError("No site weights provided.")

    if len(site_weights) != len(sample_counts):
        raise AggregationError("site_weights and sample_counts length mismatch.")

    if any(isinstance(count, bool) or not isinstance(count, Integral) or count <= 0 for count in sample_counts):
        raise AggregationError("Sample counts must be positive integers.")
    total_samples = sum(int(count) for count in sample_counts)

    if total_samples <= 0:
        raise AggregationError("Total sample count must be positive.")

    number_of_layers = len(site_weights[0])
    if number_of_layers == 0:
        raise AggregationError("Models must contain at least one tensor.")

    for weights in site_weights:
        if len(weights) != number_of_layers:
            raise AggregationError("All models must have the same number of layers.")
        for index, array in enumerate(weights):
            if array.shape != site_weights[0][index].shape:
                raise AggregationError("All tensor shapes must match.")
            if array.dtype.kind not in "fi" or array.size == 0 or not np.isfinite(array).all():
                raise AggregationError("Weights must be nonempty finite numeric arrays.")

    aggregated: list[np.ndarray] = []

    for layer_index in range(number_of_layers):
        layer_sum = None

        for weights, count in zip(site_weights, sample_counts):
            weighted_layer = weights[layer_index].astype(np.float64) * (int(count) / total_samples)

            if layer_sum is None:
                layer_sum = weighted_layer
            else:
                layer_sum = layer_sum + weighted_layer

        if not np.isfinite(layer_sum).all():
            raise AggregationError("Aggregation produced non-finite weights.")
        aggregated.append(layer_sum)

    return aggregated


def weight_delta(
    original_weights: list[np.ndarray],
    updated_weights: list[np.ndarray],
) -> list[np.ndarray]:
    if len(original_weights) != len(updated_weights):
        raise AggregationError("Original and updated weights length mismatch.")

    return [
        updated - original
        for original, updated in zip(original_weights, updated_weights)
    ]
