//////////////////////////////////////////////////////////////////////////////////////
This software was created for testing physically-based volume rendering on the GPU
//////////////////////////////////////////////////////////////////////////////////////

the software can be compiled with different parameters which can be found in the file src/Defines.h

executable folder contain the classic compilation of the program

testing-version-executable folder contain a version which try to emulate the mitsuba renderer.
It uses the same interpolation of the volume texture, camera settings and methods.

src folder contain the code

VisualStudio2015_Cuda9.0 contain the project ready to be compiled which uses VisualStudio2015 and Cuda9.0

data folder contain some scene ready to be rendered

!!!! IMPORTANT !!!!!
The software takes as first parameter the scene to render.
The type of the scene should be also provided (default is raw).
the command --help provides more informations on how to use the software
!!!!!!!!!!!!!!!!!!!!

