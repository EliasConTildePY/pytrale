"""The synthetic series trale uses for its interpolation UI previews.

Transcribed from `_msMap` in `app/lib/core/interpolation_preview.dart`. The
Dart source builds each entry as `DateTime.now().subtract(Duration(days: i))`,
so its index runs *backwards* in time; the times returned here are flipped
accordingly, otherwise every trend comes out with the wrong sign.

The six days commented out in the Dart source (2, 6, 34, 39, 57 and 58) are
deliberate gaps and are omitted here as well.
"""

import numpy as np
from numpy.typing import NDArray

_DAYS_BEFORE_TODAY: dict[int, float] = {
    0: 80.5,
    1: 79.7,
    3: 80.2,
    4: 81.4,
    5: 81.0,
    7: 80.3,
    8: 79.5,
    9: 80.1,
    10: 80.7,
    11: 80.7,
    12: 80.8,
    13: 79.7,
    14: 81.4,
    15: 80.4,
    16: 81.4,
    17: 80.3,
    18: 81.0,
    19: 81.3,
    20: 81.7,
    21: 81.9,
    22: 81.7,
    23: 81.5,
    24: 82.6,
    25: 81.7,
    26: 82.7,
    27: 81.3,
    28: 82.0,
    29: 81.1,
    30: 81.7,
    31: 81.4,
    32: 82.4,
    33: 81.2,
    35: 81.7,
    36: 82.8,
    37: 81.9,
    38: 82.7,
    40: 81.5,
    41: 82.6,
    42: 81.7,
    43: 82.7,
    44: 81.3,
    45: 80.6,
    46: 80.9,
    47: 81.5,
    48: 80.7,
    49: 80.7,
    50: 81.3,
    51: 80.5,
    52: 80.7,
    53: 82.5,
    54: 81.3,
    55: 80.5,
    56: 81.3,
    59: 82.5,
}

_SPAN_IN_DAYS = 59


def preview_measurements() -> tuple[NDArray, NDArray]:
    """Return the preview series with time running forwards.

    Returns:
        tuple[NDArray, NDArray]: Measurement times [days], ascending and
            starting at 0, and the corresponding weights [kg].
    """
    offsets = sorted(_DAYS_BEFORE_TODAY, reverse=True)
    times = np.array([float(_SPAN_IN_DAYS - i) for i in offsets])
    weights = np.array([_DAYS_BEFORE_TODAY[i] for i in offsets])
    return times, weights
