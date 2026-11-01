"""Independent targets requiring state-dependent or composed control flow."""
from __future__ import annotations


def oracle(name: str, values: list[int]) -> int:
    if name == "maximum_floor_zero":
        return max([0, *values])
    if name == "minimum_ceiling_zero":
        return min([0, *values])
    if name == "largest_magnitude":
        return max((abs(value) for value in values), default=0)
    if name == "positive_values_capped_at_two_total":
        return sum(min(value, 2) for value in values if value > 0)
    if name == "positive_count_capped_at_two":
        return min(2, sum(value > 0 for value in values))
    raise ValueError("unknown stateful programming task")


WITNESSES = {
    "maximum_floor_zero": "for if gt value acc set value end end "
                         "return acc stop",
    "minimum_ceiling_zero": "for if lt value acc set value end end "
                           "return acc stop",
    "largest_magnitude": "for if gt value acc set value end "
                         "if gt sub zero value acc set sub zero value "
                         "end end return acc stop",
    "positive_values_capped_at_two_total":
        "for if gt value zero set add acc value end "
        "if gt value two set sub acc sub value two end end return acc stop",
    "positive_count_capped_at_two":
        "for if gt value zero if lt acc two set add acc one "
        "end end end return acc stop",
}
