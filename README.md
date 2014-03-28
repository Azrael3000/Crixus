Crixus
======

Crixus is a preprocessing tool for SPH, in particular Spartacus3D, Sphynx and GPUSPH. In this document you will get some information on how to run Crixus and a description of its options.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Table of contents
=================
1. Compiling Crixus
2. Creating an STL file for Crixus
3. Running Crixus
4. Frequently encountered issues
5. Getting help


1.) Compiling Crixus
--------------------

The installation and compilation procedure is described in the file INSTALL.

2.) Creating an STL file for Crixus
-----------------------------------

Before we start a few words on how to prepare a geometry. We will use the program [SALOME](www.salome-platform.com) in the following, but any other program capable of generating _\*.stl_ files can be used.

Salome offers several options to create a geometry:

1. Create the geometry by hand in Salome
2. Import a geometry
3. Import an STL file and convert it to a geometry

The last one needs to be described in greater detail. Open a new study in Salome (File->New) and switch to the mesh module. Import an STL file (_File->Import->STL File_) and rename it to "Mesh\_import". After that load the script (_File->Load Script_) which is located in _$(CRIXUS\_PATH)/resources/_ and is called "convert\_stl\_to\_geometry.py". Next switch to the geometry module and you should be able to see a geometry called "stl\_geometry".

So by now you should have a geometry in Salome and in the next step we will create a triangle mesh. To start with that switch to the mesh module in Salome. Now create a new mesh (_Mesh->Create Mesh_). As geometry choose the geometry from the list on the left (in the above it was "stl\_geometry"). There is no need for a 3D algorithm so switch to the 2D tab. Now several options are available and it is possibly a good idea to try several of them and compare the results against each other. The options are:

1. Netgen 2D
2. Netgen 1D-2D
3. Triangle (Mephisto)

In detail:

1. **Netgen 2D**: As hypothesis choose Length From Edges, then choose a 1D algorithm as outlined below.
2. **Netgen 1D-2D**: The "Netgen 2D Simple Parameters" suffice as hypothesis. In the 1D box choose "Local Length" as option and in the 2D box tick "Length from edges".
3. **Triangle**: No hypothesis required.

For option 1 and 3 a 1D algorithm is required, to select one switch to tab 1D and choose "Wire discretization" as hypothesis choose "Max Length".

Now finally the last parameter is the characteristic length which needs to be set. Unfortunately the constraint required for Spartacus3D and Sphynx cannot be specified in any meshing tool. The constraint is as follows, the distance between a vertex particle (a triangle vertex) and a segment (located at the triangles barycentre) is allowed to be at most DR, which is the typical particle spacing. So for now I advise to take the length as approximately 3/2\*DR and then check if the distances are ok and then adapt accordingly. Since meshing usually doesn't take very long this should not be a major obstacle. Whether or not this constraint is fulfilled is checked by Crixus, as shown below, so it can be used to adapt the size.

Once the algorithms are set up, compute the mesh (Mesh->Compute) and export the resulting mesh as STL file (File->Export->STL File). Make sure that the file is saved in the **binary** STL format.

3.) Running Crixus
------------------
The syntax for launching Crixus is
```
$(CRIXUS\_BUILD_PATH)/bin/Release/Crixus path/to/file.stl DeltaR
```
where _DeltaR_ is the particle size. An example would be
```
$(CRIXUS\_BUILD_PATH)/bin/Release/Crixus spheric2.stl 0.018333
```
During the run several options are presented to the user. In the following the [second SPHERIC validation test](https://wiki.manchester.ac.uk/spheric/index.php/Test2) will be used as an example. The _spheric2.stl"_ file can be found in the _resources_ folder that is distributed as part of Crixus.

After reading the binary STL file the Crixus determines the orientation of the normals and presents the following output
```
        Normals information:
        Positive (n.(0,0,1)) minimum z: 1e+10 (0)
        Negative (n.(0,0,1)) minimum z: 0 (-1)

Swap normals (y/n):
```
The second line shows at which z level a triangle (segment) was found that has a positive dot product with the vector _(0,0,1)_. In this example it can be seen that the value is _1e+10_ which indicates that no such triangle was found. On the other hand a triangle with a negative dot product with the _(0,0,1)_ vector was found at _z = 0_. As this is identical with the bottom of the tank this indicates that the normals need to be swapped which can be done by entering _y_.

Next, is the treatment of periodicity where the following output is presented to the user
```
X-periodicity (y/n): n
Y-periodicity (y/n): n
Z-periodicity (y/n): n
```
As in this test case no periodicity is required all three inputs have been answered with _n_. Note that when using periodicity it is the task of the user to ensure that the triangle corners (vertices) on either side of the domain are exactly opposite. This is due to the fact that the vertices at the max side of the domain are removed and the segments on that side are linked to the vertices on the min side.

After the periodicity is treated, Crixus calculates the volume of the vertex particles. Next, Crixus checks whether grids for in/outflow are available. Note that this feature is disabled in the current version of Crixus as it needs further testing before it will be enabled.

The next file Crixus is looking for is *spheric2\_coarse.stl* which is a coarse version of the original STL file. This is used only in the filling algorithm and can sometimes yield improved performance in very simple geometries. Due to recent optimizations this option might be removed in future versions of the code.

More important is the file *spheric2\_fshape.stl* which again is a required to be a binary STL file. This file is optional but limits the options for filling algorithms as detailed further below.

Before the main filling starts Crixus asks the user whether he would like to limit the domain the filling algorithm works on.
```
Specify fluid container (y/n):
```
This can be particularly useful in large domains where only a small fraction will be filled with water. The main effect is that it significantly reduces the time the filling algorithm requires, which is the most computationally expensive part of Crixus. In this example we choose a slightly too large box using the following input
```
Specify fluid container (y/n): y
Specify fluid container:
Min coordinates (x,y,z): 0.0 0.0 0.0
Max coordinates (x,y,z): 1.5 1.0 0.6
```

As in our case a *spheric2\_fshape.stl* file was present two options for filling are presented
```
Choose option:
 1 ... Fluid in a box
 2 ... Fluid based on geometry
Input:
```
To illustrate both options we choose a slightly complicated approach and fill a small box with fluid first
```
Input: 1
Enter dimensions of fluid box:
xmin xmax: 0.2 0.4
ymin ymax: 0.2 0.4
zmin zmax: 0.2 0.4
Another fluid container (y/n):
```
As can be seen a fluid box has been specified that is defined by the two points *(0.2, 0.2, 0.2)* and *(0.4, 0.4, 0.4)*. The input at the end allows to choose another filling operation. As more filling is required in this case we answer with *y*. Which again lets us choose the filling algorithm. This time we choose algorithm *2* which is *Fluid based on geometry*. This algorithm then asks for a seed point as well as for a desired distance between the fluid and the wall as can be seen below
```
Input: 2
Please specify a seed point.
x, y, z = 0.5 0.5 0.5
Specify distance from fluid particles to vertex particles and segments: 0.018333
```
The seed point, from which the filling algorithm starts populating the fluid is given by *(0.5, 0.5, 0.5)* and the distance to the wall is chosen identical to the initial *DeltaR*. The filling algorithm fills all possible points that lie on a regular Cartesian grid with gridsize *DeltaR* unless it either encounters a wall or a segment of the *spheric2_fshape* file. The latter thus specifies the initial free-surface of the fluid. If this file is not present the user cannot choose this algorithm and only filling of a fluid in a box is possible.

The filling is then ended by answering *n* to the question of whether or not we would like to add another fluid container. The data for output is then prepared and the user can choose between output to VTU and H5SPH files.
```
Choose output option:
 1 ... VTU
 2 ... H5SPH
Input:
```
After the output is written to the respective file, e.g.
```
Input: 1
Writing output to file spheric2.vtu ... [OK]
```
Crixus has finished.

4.) Frequently encountered issues
---------------------------------

### Blender STL files
Currently Blender does not correctly write normals in its binary STL files. At the moment Crixus requires that the normals are defined so a warning is printed if they are not set. It is advisable to stop the computation, open the file in ParaView (or similar) and save it again as the computation most likely won't succeed otherwise.

### Support of Windows / Mac
Crixus officially only supports Linux. It might work natively under Mac but no guarantee is given.

### Running Crixus with Bumblebee
As some kernels in Crixus can have a rather long runtime Crixus requires a GPU that does not feature timeouts. This is satisfied if the GPU is not connected to a display. With bumblebee users normally run Crixus as
```
optirun path/to/Crixus ...
```
which assumes that the GPU is used for display and as such can cause Crixus to abort. To avoid this issue add the _--no-xorg_ flag to optirun, i.e.
```
optirun --no-xorg path/to/Crixus ...
```

5.) Getting help
----------------

If you need any help, found a bug or want to contribute feel free to use the [issue tracker on GitHub](https://github.com/Azrael3000/Crixus/issues) (preferred) or write me an email to firstname.initial@domain.tld where

- firstname = arno
- initial = m
- domain = gmx
- tld = at

Finally it shall be noted that the authors can be motivated by supplying them with whisky.
