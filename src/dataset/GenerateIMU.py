import xml.etree.ElementTree as ET
import numpy as np

filepath = "/Users/FranklinZhao/OpenSimProject/Simulation/Models/Rajapogal_2015/imu_data/model_default_arm_IMU.osim"

# filepath = "/Users/FranklinZhao/OpenSimProject/golf.osim"


def add_imu(component_set, imu_name, socket_frame):
    imu = ET.SubElement(component_set, "IMU", name=imu_name)
    socket_frame_element = ET.SubElement(imu, "socket_frame")
    socket_frame_element.text = socket_frame


def add_imu_frame(root, body, name, translation, rotation):
    body = root.find(f".//Body[@name='{body}']")
    components = body.find("components")

    physical_offset_frame = ET.SubElement(components, "PhysicalOffsetFrame", name=name)

    frame_geometry = ET.SubElement(
        physical_offset_frame, "FrameGeometry", name="frame_geometry"
    )

    socket_frame = ET.SubElement(frame_geometry, "socket_frame")
    socket_frame.text = ".."

    scale_factors = ET.SubElement(frame_geometry, "scale_factors")
    scale_factors.text = "0.20000000000000001 0.20000000000000001 0.20000000000000001"

    socket_parent = ET.SubElement(physical_offset_frame, "socket_parent")
    socket_parent.text = ".."

    translation_element = ET.SubElement(physical_offset_frame, "translation")
    translation_element.text = arrayToString(translation)

    orientation = ET.SubElement(physical_offset_frame, "orientation")
    orientation.text = arrayToString(rotation)


def modifyIMUTranslation(tree, body, imuName, vertical_modifier, radius):
    root = tree.getroot()

    vertical_modifiers = [
        [0, 0, 0],
        [0, vertical_modifier, 0],
        [0, -vertical_modifier, 0],
        [0, -2 * vertical_modifier, 0],
        [0, 2 * vertical_modifier, 0],
        [0, -3 * vertical_modifier, 0],
    ]

    rotations = [
        [0, 3.7699, 1.57],
        [0, 4.3982, 1.57],
        [0, 5.0265, 1.57],
        [0, 5.6549, 1.57],
        [0, 6.2832, 1.57],
    ]

    # extra rotations in degrees -> radians:
    extra_deg = [0, 30, -30]
    extra_rad = [np.deg2rad(d) for d in extra_deg]

    component_set = root.find(".//ComponentSet[@name='componentset']/objects")

    curr_position = root.find(
        f".//Body[@name='{body}']/components/PhysicalOffsetFrame[@name='{imuName}']/translation"
    )
    curr_position = parseStringArray(curr_position.text)
    curr_position[0] = 0
    curr_position = arrayToString(curr_position)

    imuNum = 1
    for m in vertical_modifiers:
        for r in rotations:
            base_position = np.array(parseStringArray(curr_position)) + np.array(m)

            # position around the body using r[1] (your azimuth)
            new_x_position = radius * np.sin(r[1])
            new_z_position = radius * np.cos(r[1])
            base_position = base_position + np.array(
                [new_x_position, 0, new_z_position]
            )

            # NOW: create 3 IMUs for this one placement
            for k, delta in enumerate(extra_rad):
                new_imu_name = f"{imuName}{imuNum+1}_{extra_deg[k]}deg"
                new_imu_frame = f"{new_imu_name}_frame"

                # apply 0/+30/-30 to ry (index 1)
                r_new = [r[0], r[1], r[2] + delta]

                add_imu_frame(root, body, new_imu_frame, base_position, r_new)
                add_imu(component_set, new_imu_name, f"/bodyset/{body}/{new_imu_frame}")

            imuNum += 1

    return tree


def parseStringArray(str):
    curr = ""
    res = []
    for c in str:
        if c == " ":
            res.append(float(curr))
            curr = ""
        else:
            curr += c
    res.append(float(curr))
    return res


def arrayToString(arr):
    return " ".join(str(f) for f in arr)


vertical_translation = 0.025
upper_radius = 0.054
lower_radius = 0.054
tree = ET.parse(filepath)
tree = modifyIMUTranslation(
    tree, "ulna_l", "ulna_l_imu", vertical_translation, lower_radius
)
tree = modifyIMUTranslation(
    tree, "ulna_r", "ulna_r_imu", vertical_translation, lower_radius
)
tree = modifyIMUTranslation(
    tree, "humerus_l", "humerus_l_imu", vertical_translation, upper_radius
)
tree = modifyIMUTranslation(
    tree, "humerus_r", "humerus_r_imu", vertical_translation, upper_radius
)
tree.write("rajapogal_half_radial_imu.osim")

# tree = ET.parse(filepath)
# tree = modifyIMUTranslation(tree, "pelvis", "pelvis_imu", 0, 0.14)
# tree.write('modified_golf.osim')
