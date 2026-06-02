"""Constants shared by the Quality Metric prototype."""

from __future__ import annotations

import numpy as np

NUM_DETECTORS = 62
NUM_ELEMENT_IDS = 201
NO_HIT_ELEMENT_ID = 0

# Existing momentum/QTracker convention masks these Python detector slots.
# These correspond to physically unused/no-data detector slots in the setup.
INACTIVE_DETECTOR_SLICES = (
    (7, 12),
    (55, 58),
    (59, 62),
)

ACTIVE_MASK = np.ones(NUM_DETECTORS, dtype=bool)
for start, stop in INACTIVE_DETECTOR_SLICES:
    ACTIVE_MASK[start:stop] = False

# Station-like detector groupings. These are used only as feature groups, not as
# authoritative detector geometry. They follow the grouping that appeared in the
# existing refinement experiments while excluding inactive slots through the
# active mask during feature calculation.
STATION_GROUPS = {
    "st1": tuple(range(0, 6)),
    "st2": tuple(range(12, 18)),
    "st3": tuple(range(18, 25)),
    "st4": tuple(range(25, 32)),
    "st5": tuple(range(32, 34)),
    "st6": tuple(range(36, 40)),
    "st7": tuple(range(44, 46)),
    "st8": tuple(range(46, 54)),
    "muplus_special": (34, 40, 42),
    "muminus_special": (35, 41, 43),
}

FRONT_DETECTORS = tuple(range(0, 18))
MIDDLE_DETECTORS = tuple(range(18, 40))
BACK_DETECTORS = tuple(range(40, 62))
TAIL_DETECTORS = tuple(range(44, 62))
