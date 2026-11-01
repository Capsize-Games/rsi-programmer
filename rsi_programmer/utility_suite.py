"""Task-owned oracles for the frozen useful-composition development suite."""
from __future__ import annotations


def oracle(name: str, values: list[int]) -> list[int]:
    """Compute specified behavior without running the synthesized program."""
    if name == "clean_positive":
        return sorted(set(value for value in values if value > 0))
    if name == "clean_nonnegative":
        return sorted(set(value for value in values if value >= 0))
    if name == "negative_magnitudes":
        return sorted(-value for value in values if value < 0)
    if name == "distinct_magnitudes_descending":
        return sorted(set(abs(value) for value in values), reverse=True)
    if name == "positive_running_total":
        total = 0
        result = []
        for value in values:
            if value > 0:
                total += value
                result.append(total)
        return result
    if name == "absolute_running_total":
        total = 0
        result = []
        for value in values:
            total += abs(value)
            result.append(total)
        return result
    if name == "distinct_positive_even":
        result = []
        for value in values:
            if value > 0 and value % 2 == 0 and value not in result:
                result.append(value)
        return result
    if name == "recent_distinct_positive":
        result = []
        for index in range(len(values) - 1, -1, -1):
            value = values[index]
            if value > 0 and value not in result:
                result.append(value)
        return result
    if name == "positive_scaled_total":
        total = 0
        result = []
        for value in values:
            if value > 0:
                total += 2 * value
                result.append(total)
        return result
    raise ValueError("unknown useful-composition task")
