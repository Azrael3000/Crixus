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

Prerequisites:

* cmake >= 2.8
* cuda
* hdf5 >= 1.8.7

Crixus uses CMake for compilation. Let us assume that you have crixus in *$(CRIXUS_PATH)* (the base path and not */path/to/crixus/src/*) and you want the building to happen in *$(CRIXUS_BUILD)* then follow the commands below:

```
mkdir $(CRIXUS_BUILD)
cd $(CRIXUS_BUILD)
cmake $(CRIXUS_PATH)
make
```

Note that it is important that *$(CRIXUS_BUILD)* and *$(CRIXUS_PATH)* are different, i.e. you should not run cmake in the main Crixus folder.

The binary is then located at *$(CRIXUS_BUILD)/bin/Release/Crixus*

Note that *"make install"* is not supported yet. To easily change the parameters of *cmake* you can use *ccmake* instead.

If hdf5 cannot be found do to lacking environmental variable you can edit the main _CMakeLists.txt_ which has a commented line that reads:
```
#set(ENV{HDF5_ROOT} "/your/path/to/hdf5")
```
Uncomment it and set the respective hdf5 path in order to use your custom installation.

2.) Creating an STL file for Crixus
-----------------------------------

Before we start a few words on how to prepare a geometry. We will use the program [SALOME](www.salome-platform.com) in the following, but any other program capable of generating _\*.stl_ files can be used.

Salome offers several options to create a geometry:

1. Create the geometry by hand in Salome
2. Import a geometry
3. Import an STL file and convert it to a geometry (Salome 7.2 and older)

With the release of Salome 7.4 the last option is no longer available. The user can now import a STL file directly in the geometry module and use this as a basis for meshing. Users of Salome 7.2 and less should read the following paragraph that details how STL files can be imported in these older versions.

Open a new study in Salome (File->New) and switch to the mesh module. Import an STL file (_File->Import->STL File_) and rename it to "Mesh\_import". After that load the script (_File->Load Script_) which is located in _$(CRIXUS\_PATH)/resources/_ and is called "convert\_stl\_to\_geometry.py". Next switch to the geometry module and you should be able to see a geometry called "stl\_geometry".

So by now you should have a geometry in Salome and in the next step we will create a triangle mesh. To start with that switch to the mesh module in Salome. Now create a new mesh (_Mesh->Create Mesh_). As geometry choose the geometry from the list on the left (in the above it was "stl\_geometry"). There is no need for a 3D algorithm so switch to the 2D tab. Now several options are available and it is possibly a good idea to try several of them and compare the results against each other. The options are:

1. Netgen 2D
2. Netgen 1D-2D
3. Triangle (Mephisto)

In detail:

1. **Netgen 2D**: As hypothesis choose Length From Edges, then choose a 1D algorithm as outlined below.
2. **Netgen 1D-2D**: The "Netgen 2D Simple Parameters" suffice as hypothesis. In the 1D box choose "Local Length" as option and in the 2D box tick "Length from edges".
3. **Triangle**: No hypothesis required.

For option 1 and 3 a 1D algorithm is required, to select one switch to tab 1D and choose "Wire discretization" as hypothesis choose "Max Size".

Now finally the last parameter is the characteristic length which needs to be set. Unfortunately the constraint required for Spartacus3D and Sphynx cannot be specified in any meshing tool. The constraint is as follows, the distance between a vertex particle (a triangle vertex) and a segment (located at the triangles barycentre) is allowed to be at most DR, which is the typical particle spacing. So for now I advise to take the length as approximately 3/2\*DR and then check if the distances are ok and then adapt accordingly. Since meshing usually doesn't take very long this should not be a major obstacle. Whether or not this constraint is fulfilled is checked by Crixus, as shown below, so it can be used to adapt the size.

Once the algorithms are set up, compute the mesh (Mesh->Compute) and export the resulting mesh as STL file (File->Export->STL File). Make sure that the file is saved in the **binary** STL format.

**Testing the triangle size:**

The triangle size for Crixus needs to fulfill a certain criterion. To check whether this is met a dedicated tool exists in the _resources_ folder of Crixus, which is called _test-triangle-size_. To compile it use
```
gcc -o test-triangle-size -lm test-triangle-size.c
```
and then run it using
```
$(CRIXUS_PATH)/resources/test-triangle-size path/to/file.stl 0.1
```
where _0.1_ is the particle size. This will tell you whether all triangles meet the criterion or not.

3.) Running Crixus
------------------
The syntax for launching Crixus is
```
$(CRIXUS_BUILD_PATH)/bin/Release/Crixus path/to/file.ini
```
Where _file_ is the name of the problem which will be referred to as $problem in the following. An example would be
```
$(CRIXUS_BUILD_PATH)/bin/Release/Crixus spheric2.ini
```
where _spheric2_ is the problem name. The file.ini is an [ini file](https://en.wikipedia.org/wiki/INI_file) which has the structure
```
[section]
option1=value1
option2=value2
; I am a comment
```
In the following the different sections that can be used will be presented. All options will be listed with there variable type and an additional comment if they are optional, including their default value.

The main section is **mesh**. Which has the following options

1. _stlfile_ (string)
2. _dr_ (float)
3. *swap\_normals* (bool, optional=false)
4. _fshape_ (string, optional $problem\_fshape.stl)
5. _zeroOpen_ (bool, optional=false)

where _stlfile_ is the path to the main stl file, _dr_ is the particle size and *swap\_normals* is an optional flag that allows the swapping of the normals of the domain, which has a default value of _false_. The _fshape_ option is the name of a STL mesh file that is used later on for filling. If the _zeroOpen_ flag is set then all vertices that are at an edge of a geometry (i.e. the one-dimensional boundary of the 2-D manifold) will have 0 mass.

In the following the [second SPHERIC validation test](https://wiki.manchester.ac.uk/spheric/index.php/Test2) will be used as an example. The _spheric2.stl_ and _spheric2.ini_ file can be found in the _resources_ folder that is distributed as part of Crixus. The **mesh** section in this case looks as follows
```
[mesh]
stlfile=spheric2.stl
dr=0.018333
swap_normals=true
fshape=spheric2_fshape.stl
```
Note that the _fshape_ option is redundant in this case as the default value ($problem\_fshape) is identical to the given value. After reading the binary STL file the Crixus determines the orientation of the normals and presents the following output
```
        Normals information:
        Positive (n.(0,0,1)) minimum z: 1e+10 (0)
        Negative (n.(0,0,1)) minimum z: 0 (-1)

```
The second line shows at which z level a triangle (segment) was found that has a positive dot product with the vector _(0,0,1)_. In this example it can be seen that the value is _1e+10_ which indicates that no such triangle was found. On the other hand a triangle with a negative dot product with the _(0,0,1)_ vector was found at _z = 0_. As this is identical with the bottom of the tank this indicates that the normals need to be swapped which was indicated by the *swap_normals=true* in the **mesh** section of the ini file.

Next, is the treatment of periodicity where the user can specify all three space directions in the **periodicity** section of the ini file. The options are

1. x (bool, optional=false)
2. y (bool, optional=false)
3. z (bool, optional=false)

As in the spheric2 test case no periodicity is required the entire **periodicity** section is missing in the ini file. Note that when using periodicity it is the task of the user to ensure that the triangle corners (vertices) on either side of the domain are exactly opposite. This is due to the fact that the vertices at the max side of the domain are removed and the segments on that side are linked to the vertices on the min side.

After the periodicity is treated, Crixus calculates the volume of the vertex particles. Next, Crixus checks whether grids for in/outflow are available. An infinite number of special boundaries can be specified and all Crixus does is set a specific value at the output "KENT" array. In GPUSPH for example this can then be used to define what type of special boundary this can be. Potential options are floating, moving or open boundaries.

There is a dedicated section for these _special boundary grids_ which is suitably calles *special_boundary_grids*. The options are

1. mesh1 (string, optional=$problem\_sbgrid_1.stl)
2. mesh2 (string, optional=$problem\_sbgrid_2.stl)
3. mesh3 (string, optional=$problem\_sbgrid_3.stl)
4. etc.

The search for special boundaries stops as soon as the respective file cannot be found. Note that in the present spheric2 test case no special boundary grid is present.

The next file Crixus is looking for is *spheric2\_coarse.stl* which is a coarse version of the original STL file. This is used only in the filling algorithm and can sometimes yield improved performance in very simple geometries. Due to recent optimizations this option might be removed in future versions of the code.

More important is the fluid shape file which was either set to a specific name in the **mesh** section using option _fshape_. It is required to be a binary STL file. This file is optional but can be used to specify the free-surface of the case.

Before the main filling starts Crixus allows the user to specify whether he would like to limit the domain the filling algorithm works on. The **fluid\_container** section defines the dimensions of the container using the following options

1. use (bool, optional=false)
2. xmin (float)
3. ymin (float)
4. zmin (float)
5. xmax (float)
6. ymax (float)
7. zmax (float)

where the _use_ option specifies whether the fluid container is activated or not and the {x,y,z}{min,max} determine the extend of the container. This can be particularly useful in large domains where only a small fraction will be filled with water. The main effect is that it significantly reduces the time the filling algorithm requires, which is the most computationally expensive part of Crixus. In this example we choose a slightly too large box using the following input
```
[fluid_container]
use=true
xmin=0.0
ymin=0.0
zmin=0.0
xmax=1.5
ymax=1.0
zmax=0.6
```
Next comes the main filling of the geometry. The algorithms can be called as many times as required and each call to the algorithm must be placed in a separate section. They are named *fill\_0*, *fill\_1*, *fill\_2*, etc. and all numbers starting from 0 need to be present. The first option in each *fill\_n* is titled _option_ and has a default value of "box", the only other option currently is "geometry". Depending on this first option the other options are as follows:

1. option=box
2. xmin (float)
3. ymin (float)
4. zmin (float)
5. xmax (float)
6. ymax (float)
7. zmax (float)

Where {x,y,z}{min,max} indicate the size of the box of fluid that will be filled. To illustrate both options we choose a slightly complicated approach and fill a small box with fluid first
```
[fill_0]
option=box
xmin=0.2
ymin=0.2
zmin=0.2
xmax=0.4
ymax=0.4
zmax=0.4
```
As can be seen a fluid box has been specified that is defined by the two points *(0.2, 0.2, 0.2)* and *(0.4, 0.4, 0.4)*. In order to call the filling algorithm a second time a **fill_1** section is also specified using the *geometry* option. This algorithm takes a seed point as input as well as a desired distance between the fluid and the wall resulting in the following options

1. option=geometry
2. xseed (float)
3. yseed (float)
4. zseed (float)
5. dr\_wall (float, optional=dr)

In the Spheric2 test case that is used an example this results in the following output
```
[fill_1]
option=geometry
xseed=0.5
yseed=0.5
zseed=0.5
dr_wall=0.018333
```
The seed point, from which the filling algorithm starts populating the fluid is given by *(0.5, 0.5, 0.5)* and the distance to the wall is chosen identical to the initial *dr*. Note that due to the default value of *dr_wall* being equal to *dr* we would not have had to specify this option. The filling algorithm fills all possible points that lie on a regular Cartesian grid with gridsize *dr* unless it either encounters a wall or a segment of the *spheric2_fshape.stl* file. The latter thus specifies the initial free-surface of the fluid.

The filling is then completed as no **fill_2** section is present. The data for output is then prepared and the user can choose between output to VTU and H5SPH files. This is done by specifying a **output** section which has the options

1. format (string, optional=vtu)
2. name (string, optional=$problem)
3. split (bool, optional=false)

where the format option can currently be either *vtu* or *h5sph*. The name can be an optional name for the output file, if it is not set the standard $problem name will be used. If you are using GPUSPH then you need to choose h5sph as it is the only currently supported format. The *split* option causes not one file to be written but multiple ones, i.e. one for each special boundary grid.
```
[output]
format=h5sph
name=spheric2_ready_to_run
```
After the output is written to the file (in the example case: *0.spheric2\_ready\_to\_run.h5sph*) and after that Crixus has finished.

4.) Frequently encountered issues
---------------------------------

### Chosing a different GPU
For large cases sufficient memory on the GPU is required. Crixus normally chooses the first suitable GPU to do the computation on. However, in the *ini* file a specific GPU can be specified with:
```
[system]
gpu-id=n
```
where *n* is an integer that specifies the appropriate GPU index. This index can be obtained by running the *nvidia-smi -L* command.

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
