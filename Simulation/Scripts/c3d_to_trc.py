import opensim as osim
inputC3d = "/Users/FranklinZhao/OpenSimProject/Simulation/Models/Rajapogal_2015/marker_data/SN001/SN001/SN001_0026_tug_03.c3d"
outputTRC = "/Users/FranklinZhao/OpenSimProject/Simulation/Models/Rajapogal_2015/marker_data/trc/SN001_0026_tug_03.trc"


# convert c3d file into marker and forces data tables
c3dFileAdapter = osim.C3DFileAdapter()
c3dFileAdapter.setLocationForForceExpression(osim.C3DFileAdapter.ForceLocation_CenterOfPressure);
tables = c3dFileAdapter.read(inputC3d)
markersTable = c3dFileAdapter.getMarkersTable(tables)
forcesTable = c3dFileAdapter.getForcesTable(tables)

# convert marker and forces data tables in .trc files
trcFileAdapter = osim.TRCFileAdapter()
trcFileAdapter.write(markersTable, outputTRC)


