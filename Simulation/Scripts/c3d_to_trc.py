import opensim as osim
inputC3d = "/home/franklin/Research/Simulation/Models/Rajapogal_2015/marker_data/SN001/SN001/SN001_0028_towel_R01.c3d"
outputTRC = "/Users/FranklinZhao/OpenSimProject/Simulation/Models/Rajapogal_2015/marker_data/trc/SN001_0028_towel_R01.trc"
# inputC3d = "/Users/FranklinZhao/Research/Simulation/Models/Rajapogal_2015/marker_data/SN001/SN001/SN001_0042_back_R01.c3d"
# outputTRC = "/Users/FranklinZhao/Research/Simulation/Models/Rajapogal_2015/marker_data/trc/SN001_0042_back_R01.trc"


# convert c3d file into marker and forces data tables
c3dFileAdapter = osim.C3DFileAdapter()
c3dFileAdapter.setLocationForForceExpression(osim.C3DFileAdapter.ForceLocation_CenterOfPressure);
tables = c3dFileAdapter.read(inputC3d)
markersTable = c3dFileAdapter.getMarkersTable(tables)
forcesTable = c3dFileAdapter.getForcesTable(tables)

# convert marker and forces data tables in .trc files
trcFileAdapter = osim.TRCFileAdapter()
trcFileAdapter.write(markersTable, outputTRC)


