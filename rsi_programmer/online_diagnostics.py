"""Read-only diagnostics for the cold initial online-policy state."""
from __future__ import annotations

import copy
import math
from typing import Any

from rsi_programmer.online_spiking import (
    ACTION_ORDER,
    HIDDEN_COUNT,
    MAX_WEIGHT,
    OnlineSpikingPolicy,
)

INITIAL_FEATURES = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def inspect_policy(
    policy: OnlineSpikingPolicy | None,
) -> dict[str, Any]:
    """Describe initial action probabilities without mutating ``policy``.

    For a policy, this evaluates a cold membrane using the fixed initial
    context, the full action mask, and the current readout. It does not draw
    from the policy RNG or expose learned weight arrays. ``None`` represents
    fixed grammar streaming, whose first lane is always ``add``.
    """
    if policy is None:
        probabilities = [1.0, 0.0, 0.0]
        logits = None
        baseline = None
        readout_values: list[float] = []
    else:
        clone = copy.copy(policy)
        clone._membrane = [0.0] * HIDDEN_COUNT
        features = clone._validate_features(INITIAL_FEATURES)
        rates, _spikes = clone._encode(features)
        logits = [
            sum(weight * rate
                for weight, rate in zip(policy._weights[index], rates))
            + policy._bias[index]
            for index in range(len(ACTION_ORDER))
        ]
        probabilities_by_index = clone._masked_softmax(
            logits, tuple(range(len(ACTION_ORDER)))
        )
        probabilities = [probabilities_by_index[index]
                          for index in range(len(ACTION_ORDER))]
        baseline = float(policy._baseline)
        readout_values = [value for row in policy._weights for value in row]
        readout_values.extend(policy._bias)

    entropy = -sum(value * math.log(value)
                   for value in probabilities if value > 0.0)
    return {
        "logits": logits,
        "probabilities": probabilities,
        "entropy_nats": entropy,
        "max_probability": max(probabilities),
        "readout_l2": math.sqrt(sum(value * value
                                     for value in readout_values)),
        "clipped_weights": sum(
            abs(value) >= MAX_WEIGHT for value in readout_values
        ),
        "baseline": baseline,
    }
