from smesh import *
import salome
SetCurrentStudy(salome.myStudy)

#stlFile = "path/mesh.stl"
#stlMesh = CreateMeshesFromSTL(stlFile)
meshname="Mesh_import"
stlMesh = salome.myStudy.FindObject(meshname).GetObject()

# Make geometry from stlMesh
# 1) vertices
vertices = []
for n in stlMesh.GetNodesId():
    x,y,z = stlMesh.GetNodeXYZ( n )
    vertices.append( geompy.MakeVertex( x,y,z ))
# 2) faces
faces = []
for tria in stlMesh.GetElementsByType( SMESH.FACE ):
    n1,n2,n3 = stlMesh.GetElemNodes( tria )
    wire = geompy.MakePolyline( [ vertices[n1-1], vertices[n2-1], vertices[n3-1]], theIsClosed=1)
    faces.append( geompy.MakeFace( wire, isPlanarWanted=True ))
# 3) shell
shell = geompy.MakeShell( faces )
# 4) heal the shell
shape = geompy.MakeGlueEdges( shell, 1e-7 )
geompy.addToStudy( shape, "stl_geometry" )
