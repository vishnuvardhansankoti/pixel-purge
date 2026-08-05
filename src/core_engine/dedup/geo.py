"""Great-circle distance helpers (Appendix A)."""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

_EARTH_RADIUS_M = 6_371_000.0
_METERS_PER_MILE = 1609.344


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in meters between two WGS84 coordinates."""
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_M * asin(sqrt(a))


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return haversine(lat1, lon1, lat2, lon2) / _METERS_PER_MILE
