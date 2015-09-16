## How to build the program
Needed version of the CUDA SDK: 5.0 or higher.
Also, the OpenCV libraries have to be available.

# Build under Windows with Visual Studio 2013
Simply open the SuperResolution.sln in the VisualProject directory with Visual Studio 2013.
Then, in Visual Studio, go to Build -> Build Solution.

# Builder under Linux with CMake
In the repository root directory execute the following commands:
```sh
mkdir build
cd build
cmake ..
make
```
The binary can then be found in the build directory
