import opensim as osim
import numpy

modelFileName = "/Users/FranklinZhao/OpenSimProject/Simulation/Models/gait2354/scale_data/gait2354_scaled.osim"
model = osim.Model(modelFileName)

markerFileName = "/Users/FranklinZhao/OpenSimProject/Simulation/Models/gait2354/marker_data/subject01_walk1.trc"
outputMotFileName = "/Users/FranklinZhao/OpenSimProject/Simulation/Models/gait2354/inverse_kinematics_data/subject01_walk1_ik.mot"

ikTool = osim.InverseKinematicsTool()
ikTastkSet = osim.IKTaskSet("/Users/FranklinZhao/OpenSimProject/Simulation/Models/gait2354/inverse_kinematics_data/ik_Task_Set.xml")

ikTool.setModel(model)
ikTool.set_marker_file(markerFileName)
ikTool.set_IKTaskSet(ikTastkSet)
ikTool.set_output_motion_file(outputMotFileName)
ikTool.set_constraint_weight(20)
ikTool.set_accuracy(0.00001)
ikTool.set_time_range(0, 1)
ikTool.set_time_range(1, 2)

ikTool.run()

