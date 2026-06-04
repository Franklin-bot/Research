from collections import OrderedDict

import numpy as np

SEGMENT_IMUS = [
    "humerus_r_imu",
    "humerus_l_imu",
    "ulna_r_imu",
    "ulna_l_imu",
]
LOCATION_NAMES = ["default", "left", "right", "up", "down"]
KINEMATICS_COLS = [
    "lumbar_extension",
    "lumbar_bending",
    "lumbar_rotation",
    "arm_flex_r",
    "arm_add_r",
    "arm_rot_r",
    "elbow_flex_r",
    "pro_sup_r",
    "arm_flex_l",
    "arm_add_l",
    "arm_rot_l",
    "elbow_flex_l",
    "pro_sup_l",
]
MASK_VALUE = -2.0

LOCATION_PLACEMENT_SUFFIXES = OrderedDict(
    [
        ("default", ""),
        ("left", "2_0deg"),
        ("right", "6_0deg"),
        ("up", "9_0deg"),
        ("down", "14_0deg"),
    ]
)


def build_virtual_imu_names(segment_imus=None):
    segment_imus = segment_imus or SEGMENT_IMUS
    names_by_location = OrderedDict()

    for location, suffix in LOCATION_PLACEMENT_SUFFIXES.items():
        if suffix:
            names_by_location[location] = [
                f"{segment_imu}{suffix}" for segment_imu in segment_imus
            ]
        else:
            names_by_location[location] = list(segment_imus)

    return names_by_location


def flatten_virtual_imu_names(names_by_location=None):
    names_by_location = names_by_location or build_virtual_imu_names()
    return [
        imu_name
        for location in LOCATION_NAMES
        for imu_name in names_by_location[location]
    ]


def build_location_column_indices(columns, names_by_location=None):
    names_by_location = names_by_location or build_virtual_imu_names()
    indices_by_location = OrderedDict()

    for location in LOCATION_NAMES:
        indices = []
        for index, column in enumerate(columns):
            if any(
                column.startswith(f"{imu_name}_")
                for imu_name in names_by_location[location]
            ):
                indices.append(index)
        indices_by_location[location] = indices

    return indices_by_location


def mask_except_location(
    data,
    columns,
    indices_by_location,
    retained_location,
    kin_dim=len(KINEMATICS_COLS),
    mask_value=MASK_VALUE,
):
    if retained_location not in indices_by_location:
        raise ValueError(
            f"Unknown retained IMU location {retained_location!r}. "
            f"Expected one of: {', '.join(indices_by_location)}"
        )

    masked = np.array(data, copy=True)
    masked[:, :kin_dim] = mask_value

    retained = set(indices_by_location[retained_location])
    for index in range(kin_dim, len(columns)):
        if index not in retained:
            masked[:, index] = mask_value

    return masked
