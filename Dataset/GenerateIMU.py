import xml.etree.ElementTree as ET
import numpy as np

filepath = "/Users/FranklinZhao/OpenSimProject/Simulation/Models/Rajapogal_2015/imu_data/model_default_arm_IMU.osim"

# filepath = "/Users/FranklinZhao/OpenSimProject/golf.osim"

def add_imu(component_set, imu_name, socket_frame):
    imu = ET.SubElement(component_set, 'IMU', name=imu_name)
    socket_frame_element = ET.SubElement(imu, 'socket_frame')
    socket_frame_element.text = socket_frame

def add_imu_frame(root, body, name, translation, rotation):
    body = root.find(f".//Body[@name=\'{body}\']")
    components = body.find('components')

    physical_offset_frame = ET.SubElement(components, 'PhysicalOffsetFrame', name=name)
    
    frame_geometry = ET.SubElement(physical_offset_frame, 'FrameGeometry', name='frame_geometry')
    
    socket_frame = ET.SubElement(frame_geometry, 'socket_frame')
    socket_frame.text = '..'
    
    scale_factors = ET.SubElement(frame_geometry, 'scale_factors')
    scale_factors.text = '0.20000000000000001 0.20000000000000001 0.20000000000000001'
    
    socket_parent = ET.SubElement(physical_offset_frame, 'socket_parent')
    socket_parent.text = '..'
    
    translation_element = ET.SubElement(physical_offset_frame, 'translation')
    translation_element.text = arrayToString(translation)
    
    orientation = ET.SubElement(physical_offset_frame, 'orientation')
    orientation.text = arrayToString(rotation)

def modifyIMUTranslation(tree, body, imuName, vertical_modifier, radius):
    root = tree.getroot()
    vertical_modifiers = [[0, 0, 0,], [0, vertical_modifier, 0], [0, -vertical_modifier, 0],[0, 2 * vertical_modifier, 0], [0, -2 * vertical_modifier, 0]]
    # 1/16ths
    rotations = [[0, 0, 1.57], [0, 0.393, 1.57], [0, 0.785, 1.57], [0, 1.178, 1.57], [0, 1.571, 1.57], [0, 1.961, 1.57], [0, 2.356, 1.57], [0, 2.7489, 1.57], [0, 3.1416, 1.57], [0, 3.5343, 1.57],\
           [0, 3.927, 1.57], [0, 4.3197, 1.57], [0, 4.7124, 1.57], [0, 5.1051, 1.57], [0, 5.4978, 1.57], [0, 5.8905, 1.57]]

    # 1/8ths
    # rotations = [[0, 0, 1.57], [0, 0.785, 1.57], [0, 1.571, 1.57], [0, 2.356, 1.57], [0, 3.1416, 1.57], [0, 3.927, 1.57], [0, 4.7124, 1.57], [0, 5.4978, 1.57]]
    component_set = root.find(".//ComponentSet[@name='componentset']/objects")

    curr_position = root.find(f".//Body[@name='{body}']/components/PhysicalOffsetFrame[@name='{imuName}']/translation")
    curr_position = curr_position.text
    curr_position = parseStringArray(curr_position)
    curr_position[0] = 0
    curr_position = arrayToString(curr_position)

    # curr_position = [-radius/2, 0, 0]
    # curr_position = arrayToString(curr_position)


# augment vertically and horizontally
    imuNum = 1
    for m in vertical_modifiers:
        for r in rotations:
            new_imu_name = f"{imuName}{imuNum+1}"
            new_imu_position = np.array(parseStringArray(curr_position)) + np.array(m)
            new_x_position = radius * np.sin(r[1])
            new_z_position = radius * np.cos(r[1])
            new_imu_position += np.array([new_x_position, 0, new_z_position])
            new_imu_frame = f"{new_imu_name}_frame"
            add_imu_frame(root, body, new_imu_frame, new_imu_position, r)
            add_imu(component_set, new_imu_name, f'/bodyset/{body}/'+new_imu_name+'_frame')
            imuNum += 1
            print(r)

# only augment horizontally
    # imuNum = 1
    # for r in rotations:
    #     new_imu_name = f"{imuName}{imuNum+1}"
    #     new_imu_position = np.array(parseStringArray(curr_position))
    #     new_x_position = radius * np.sin(r[1])
    #     new_z_position = radius * np.cos(r[1])
    #     new_imu_position += np.array([new_x_position, 0, new_z_position])
    #     new_imu_frame = f"{new_imu_name}_frame"
    #     add_imu_frame(root, body, new_imu_frame, new_imu_position, r)
    #     add_imu(component_set, new_imu_name, f'/bodyset/{body}/'+new_imu_name+'_frame')
    #     imuNum += 1
    #     print(r)


# Save the modified XML to a new file
    return tree
    

def parseStringArray(str):
    curr = ""
    res = []
    for c in str:
        if c == ' ':
            res.append(float(curr))
            curr = ""
        else:
            curr+=c
    res.append(float(curr))
    return res

def arrayToString(arr):
    return " ".join(str(f) for f in arr)


vertical_translation = 0.025
upper_radius = 0.054
lower_radius = 0.054 
tree = ET.parse(filepath)
tree = modifyIMUTranslation(tree, "ulna_l", "ulna_l_imu", vertical_translation, lower_radius)
tree = modifyIMUTranslation(tree, "ulna_r", "ulna_r_imu", vertical_translation, lower_radius)
tree = modifyIMUTranslation(tree, "humerus_l", "humerus_l_imu", vertical_translation, upper_radius)
tree = modifyIMUTranslation(tree, "humerus_r", "humerus_r_imu", vertical_translation, upper_radius)
tree.write('rajapogal_synthetic_imu.osim')

# tree = ET.parse(filepath)
# tree = modifyIMUTranslation(tree, "pelvis", "pelvis_imu", 0, 0.14)
# tree.write('modified_golf.osim')







