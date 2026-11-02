"""Development workloads and independent list-processing specifications."""
from __future__ import annotations

TASKS = ("clip_negative_readings", "cap_positive_readings",
         "negative_magnitudes")
CONTROL = {
    "clip_negative_readings": tuple((
        "for if gt value zero append value end "
        "if le value zero append zero end end return output stop").split()),
    "cap_positive_readings": tuple((
        "for if gt value zero if le value two append value end end "
        "if gt value two append two end end return output stop").split()),
    "negative_magnitudes": tuple((
        "for if lt value zero append sub zero value end end "
        "return output stop").split()),
    "duplicate_readings": tuple((
        "for append value append value end return output stop").split()),
}


def oracle(name: str, values: list[int]) -> list[int]:
    if name == "clip_negative_readings":
        return [max(0, value) for value in values]
    if name == "cap_positive_readings":
        return [min(2, value) for value in values if value > 0]
    if name == "negative_magnitudes":
        return [-value for value in values if value < 0]
    if name == "duplicate_readings":
        return [value for item in values for value in (item, item)]
    raise ValueError(f"unknown array workload {name!r}")
