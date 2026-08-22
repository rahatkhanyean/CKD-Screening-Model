"""Parsing of the pre-discretised interval strings used throughout this dataset.

Why this module exists
----------------------
``ckd-dataset-v2.csv`` does **not** contain raw laboratory values. Every
continuous variable was discretised by the data publishers into interval
labels before release, e.g.::

    sg   -> "< 1.007", "1.009 - 1.011", ..., ">= 1.023"
    bgr  -> "< 112", "112 - 154", ..., ">= 448"
    age  -> "< 12", "12 - 20", ..., ">= 74"

The original continuous measurements are therefore **not recoverable**. Any
analysis of this file inherits that information loss; it is a property of the
released data, not a modelling choice.

Encoding strategy (leakage-proof by construction)
-------------------------------------------------
Each interval label is mapped to a single *representative numeric value*:

* closed interval ``"a - b"``      -> midpoint ``(a + b) / 2``
* open lower tail ``"< a"``        -> ``a``   (the only finite edge available)
* open upper tail ``">= z"``       -> ``z``   (the only finite edge available)

This mapping is a **pure function of the individual cell string**. It uses no
information from any other row, from other patients, or from the outcome. It
therefore cannot leak information across cross-validation folds, which is why
it is applied before splitting rather than being "fitted" inside folds. Every
other transformation (imputation, scaling, selection, calibration) *is*
fold-local.

The representative values are strictly increasing across the ordered bins of
every feature in this dataset except for the documented degenerate ``su`` bins
(see :func:`ckd.data.quality.audit_bin_monotonicity`), so the encoding preserves
the ordinal information that survived discretisation, while keeping the values
on their original clinical scale (mg/dL, g/dL, years, ...) for interpretability.

Limitation, stated explicitly: open-ended tail bins are represented by their
finite edge, which compresses the tails. ``sc < 3.65`` for example covers the
entire normal range plus mild impairment and is represented by 3.65.
"""

from __future__ import annotations

import math
import re
from typing import Final

# Unicode forms that appear in the released file, plus ASCII fallbacks.
_GE_TOKENS: Final = ("\u2265", ">=", "\u2a7e")
_LE_TOKENS: Final = ("\u2264", "<=", "\u2a7d")

# "1.019 - 1.021", "112 - 154", "-3 - -1" (negative-safe), en-dash tolerated.
_RANGE_RE: Final = re.compile(
    r"^\s*(?P<lo>[+-]?\d+(?:\.\d+)?)\s*[-\u2013\u2014]\s*(?P<hi>[+-]?\d+(?:\.\d+)?)\s*$"
)
_NUMBER_RE: Final = re.compile(r"^\s*[+-]?\d+(?:\.\d+)?\s*$")


class BinParseError(ValueError):
    """Raised when a cell cannot be interpreted as a bin label or a number."""


def normalise_label(value: object) -> str:
    """Collapse whitespace and unify comparison operators in a raw cell."""
    text = str(value)
    text = text.replace("\u00a0", " ")
    for tok in _GE_TOKENS:
        text = text.replace(tok, ">=")
    for tok in _LE_TOKENS:
        text = text.replace(tok, "<=")
    return " ".join(text.split())


def parse_bin(value: object) -> tuple[float, float, float]:
    """Return ``(lower, upper, representative)`` for one interval label.

    ``lower``/``upper`` may be ``-inf``/``+inf`` for open-ended bins. The
    representative value is finite in every case.

    Raises
    ------
    BinParseError
        If the label is not a recognised interval or bare number.
    """
    text = normalise_label(value)
    if not text:
        raise BinParseError("empty label")

    m = _RANGE_RE.match(text)
    if m:
        lo = float(m.group("lo"))
        hi = float(m.group("hi"))
        if hi < lo:  # tolerate a reversed label rather than silently mis-ordering
            lo, hi = hi, lo
        return lo, hi, (lo + hi) / 2.0

    if text.startswith(">="):
        z = float(text[2:].strip())
        return z, math.inf, z
    if text.startswith(">"):
        z = float(text[1:].strip())
        return z, math.inf, z
    if text.startswith("<="):
        a = float(text[2:].strip())
        return -math.inf, a, a
    if text.startswith("<"):
        a = float(text[1:].strip())
        return -math.inf, a, a

    if _NUMBER_RE.match(text):
        v = float(text)
        return v, v, v

    raise BinParseError(f"unrecognised bin label: {value!r}")


def representative(value: object) -> float:
    """Representative numeric value of one interval label (see module docstring)."""
    return parse_bin(value)[2]


def bin_bounds(value: object) -> tuple[float, float]:
    """``(lower, upper)`` bounds of one interval label."""
    lo, hi, _ = parse_bin(value)
    return lo, hi


def is_bin_label(value: object) -> bool:
    """True when ``value`` can be parsed as an interval label or number."""
    try:
        parse_bin(value)
    except (BinParseError, ValueError):
        return False
    return True
