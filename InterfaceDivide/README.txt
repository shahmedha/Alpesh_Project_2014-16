********************************************************************************

2) CCM (CUDA Content-Based Matching) runs on CUDA GPUs. It requires requires
CUDA drivers, CUDA toolkit (version 3.0 or greater), and a cuda-capable GPU to
run.

If you are running on a 32 bit machine, remove the -m64 option in Makefile to
compile correctly

Check the compute capabilities of your GPU and modify the -arch option in
Makefile accordingly.  Use -arch sm_11 if you are not sure of the capabilities
of your hardware.

Download and expand CCM.tar.gz
make all to compile
make clean to remove all compiled objects
./CCM 1 to run

********************************************************************************

Detailed descriptions and evaluations of CCM and OCM can be found in the
following papers:

"High Performance Content-Based Matching Using GPUs"
by Alessandro Margara and Gianpaolo Cugola
http://home.dei.polimi.it/margara/Papers/cudaFF.pdf

"High Performance Content-Based Matching Using Off-The-Shelves Parallel
Hardware"
by Alessandro Margara and Gianpaolo Cugola
http://home.dei.polimi.it/margara/Papers/PCM.pdf

Only Algorithm 1 is correctly tested.
