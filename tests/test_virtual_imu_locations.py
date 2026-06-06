import unittest

import numpy as np

from src.ML.VirtualIMULocations import (
    KINEMATICS_COLS,
    LOCATION_NAMES,
    build_location_column_indices,
    build_virtual_imu_names,
    mask_except_location,
)


class VirtualIMULocationTests(unittest.TestCase):
    def test_builds_five_non_rotated_locations_for_each_segment(self):
        names = build_virtual_imu_names()

        self.assertEqual(list(names), LOCATION_NAMES)
        self.assertEqual(
            names["default"],
            [
                "humerus_r_imu",
                "humerus_l_imu",
                "ulna_r_imu",
                "ulna_l_imu",
            ],
        )
        self.assertEqual(
            names["left"],
            [
                "humerus_r_imu2_0deg",
                "humerus_l_imu2_0deg",
                "ulna_r_imu2_0deg",
                "ulna_l_imu2_0deg",
            ],
        )
        self.assertEqual(
            names["right"],
            [
                "humerus_r_imu6_0deg",
                "humerus_l_imu6_0deg",
                "ulna_r_imu6_0deg",
                "ulna_l_imu6_0deg",
            ],
        )
        self.assertEqual(
            names["up"],
            [
                "humerus_r_imu9_0deg",
                "humerus_l_imu9_0deg",
                "ulna_r_imu9_0deg",
                "ulna_l_imu9_0deg",
            ],
        )
        self.assertEqual(
            names["down"],
            [
                "humerus_r_imu14_0deg",
                "humerus_l_imu14_0deg",
                "ulna_r_imu14_0deg",
                "ulna_l_imu14_0deg",
            ],
        )

    def test_masks_kinematics_and_all_but_retained_location_for_both_times(self):
        imu_names_by_location = {
            "default": ["seg_imu"],
            "left": ["seg_imu2_0deg"],
            "right": ["seg_imu6_0deg"],
            "up": ["seg_imu9_0deg"],
            "down": ["seg_imu14_0deg"],
        }
        columns = list(KINEMATICS_COLS)
        for time_suffix in ("t", "tminus1"):
            for imu_name in sum(imu_names_by_location.values(), []):
                columns.extend(
                    [
                        f"{imu_name}_qw_{time_suffix}",
                        f"{imu_name}_vx_{time_suffix}",
                    ]
                )
        data = np.arange(2 * len(columns), dtype=np.float64).reshape(2, len(columns))

        indices = build_location_column_indices(columns, imu_names_by_location)
        masked = mask_except_location(data, columns, indices, "up")

        self.assertTrue(np.all(masked[:, : len(KINEMATICS_COLS)] == -2.0))
        for index, column in enumerate(
            columns[len(KINEMATICS_COLS) :],
            start=len(KINEMATICS_COLS),
        ):
            if column.startswith("seg_imu9_0deg_"):
                np.testing.assert_array_equal(masked[:, index], data[:, index])
            else:
                self.assertTrue(np.all(masked[:, index] == -2.0), column)


if __name__ == "__main__":
    unittest.main()
