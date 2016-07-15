// 
// Adapted from Paulius Micikevicius (pauliusm@nvidia.com)
//

#include <stdio.h>
#include <stdlib.h>

#define	NUM_GPUS	2

void process_error( const cudaError_t &error, char *string=0, bool verbose=false )
{
	if( error != cudaSuccess || verbose )
	{
		if( string )
			printf( string );
		printf( ": %s\n", cudaGetErrorString( error ) );
	}

	if( error != cudaSuccess )
		exit(-1);
}

int main( int argc, char *argv[] )
{
	size_t num_bytes = 16*1024*1024;
	int nreps = 10;
	int gpu_0 = 0;
	int gpu_1 = 1;

	if( argc >= 2 )
		num_bytes = (size_t)( atoi( argv[1] ) * 1024*1024 );
	if( argc >= 3 )
		nreps = atoi( argv[2] );
	if( argc >= 4 )
		gpu_0 = atoi( argv[3] );
	if( argc >= 5 )
		gpu_1 = atoi( argv[4] );

	cudaError_t error = cudaSuccess;
	
	cudaDeviceProp gpu_prop;
	cudaGetDeviceProperties( &gpu_prop, gpu_0 );
	printf("GPU 0: %s\n", gpu_prop.name  );
	cudaGetDeviceProperties( &gpu_prop, gpu_1 );
	printf("GPU 1: %s\n", gpu_prop.name  );
	
	void *d_a[4] = {0, 0, 0, 0};
	void *d_b[4] = {0, 0, 0, 0};

	cudaSetDevice( gpu_0 );
	error = cudaMalloc( &d_a[0], num_bytes );
	process_error( error, "allocate a on GPU 0" );
	error = cudaMalloc( &d_b[0], num_bytes );
	process_error( error, "allocate b on GPU 0" );
	error = cudaDeviceEnablePeerAccess( gpu_1, 0 );
	process_error( error, "enable GPU 0 to access GPU 1's memory" );

	cudaSetDevice( gpu_1 );
	error = cudaMalloc( &d_a[1], num_bytes );
	process_error( error, "allocate a on GPU 1" );
	error = cudaMalloc( &d_b[1], num_bytes );
	process_error( error, "allocate b on GPU 1" );
	error = cudaDeviceEnablePeerAccess( gpu_0, 0 );
	process_error( error, "enable GPU 1 to access GPU 0's memory" );
	
	cudaSetDevice( gpu_0 );

	float elapsed_time_ms = 0.f;
	float throughput_gbs  = 0.f;
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	cudaStream_t stream_on_gpu_0, stream_on_gpu_1;

        cudaSetDevice( gpu_0 );
        cudaStreamCreate( &stream_on_gpu_0 );
        cudaSetDevice( gpu_1 );
        cudaStreamCreate( &stream_on_gpu_1 );

	cudaSetDevice( gpu_0 );

	///////////////////////////
	// pull copy
	//
	cudaEventRecord( start, 0 );
	for( int i=0; i<nreps; i++ ) {
		cudaMemcpyPeerAsync( d_b[0], gpu_0, d_b[1], gpu_1, num_bytes, stream_on_gpu_0 );
	}
	error = cudaStreamSynchronize(stream_on_gpu_0);
	cudaEventRecord( stop, 0 );
	error = cudaDeviceSynchronize();
	process_error( error, "sync after pull copy" );
		
	error = cudaEventElapsedTime( &elapsed_time_ms, start, stop );
	process_error( error, "get event elapsed time" );
	elapsed_time_ms /= nreps;
	throughput_gbs = num_bytes * 1e-6f / elapsed_time_ms;
	printf( "%d ->%d: %7.2f %7.2f\n", gpu_1, gpu_0, elapsed_time_ms, throughput_gbs );

	///////////////////////////
	// push copy
	//
	cudaEventRecord( start, 0 );
	for( int i=0; i<nreps; i++ ) {
		cudaMemcpyPeerAsync( d_a[1], gpu_1, d_a[0], gpu_0, num_bytes, stream_on_gpu_0 );
	}
	cudaEventRecord( stop, 0 );
	error = cudaDeviceSynchronize();
	process_error( error, "sync after push copy" );
		
	error = cudaEventElapsedTime( &elapsed_time_ms, start, stop );
	process_error( error, "get event elapsed time" );
	elapsed_time_ms /= nreps;
	throughput_gbs = num_bytes * 1e-6f / elapsed_time_ms;
	printf( "%d ->%d: %7.2f %7.2f\n", gpu_0, gpu_1, elapsed_time_ms, throughput_gbs );


	///////////////////////////
	// exchange with sync
	//
	cudaEventRecord( start, 0 );
	for( int i=0; i<nreps; i++ ) {
		cudaMemcpyPeerAsync( d_a[1], gpu_1, d_a[0], gpu_0, num_bytes, stream_on_gpu_0 );
		cudaMemcpyPeerAsync( d_b[0], gpu_0, d_b[1], gpu_1, num_bytes, stream_on_gpu_1 );
		cudaDeviceSynchronize();
	}
	cudaEventRecord( stop, 0 );
	error = cudaDeviceSynchronize();
	process_error( error, "sync after exchange" );
		
	error = cudaEventElapsedTime( &elapsed_time_ms, start, stop );
	process_error( error, "get event elapsed time" );
	elapsed_time_ms /= nreps;
	throughput_gbs = num_bytes * 2e-6f / elapsed_time_ms;
	printf( "%d<->%d: %7.2f %7.2f\n", gpu_0, gpu_1, elapsed_time_ms, throughput_gbs );

	///////////////////////////
	// exchange without sync
	//
	cudaEventRecord( start, 0 );
	for( int i=0; i<nreps; i++ ) {
		cudaMemcpyPeerAsync( d_a[1], gpu_1, d_a[0], gpu_0, num_bytes, stream_on_gpu_0 );
		cudaMemcpyPeerAsync( d_b[0], gpu_0, d_b[1], gpu_1, num_bytes, stream_on_gpu_1 );
	}
	cudaEventRecord( stop, 0 );
	error = cudaDeviceSynchronize();
	process_error( error, "sync after exchange" );
		
	error = cudaEventElapsedTime( &elapsed_time_ms, start, stop );
	process_error( error, "get event elapsed time" );
	elapsed_time_ms /= nreps;
	throughput_gbs = num_bytes * 2e-6f / elapsed_time_ms;
	printf( "%d<->%d: %7.2f %7.2f\n", gpu_0, gpu_1, elapsed_time_ms, throughput_gbs );


	
	cudaSetDevice( gpu_0 );
	error = cudaFree( d_a[0] );
	process_error( error, "free memory on GPU 0" );
	error = cudaDeviceReset();
	process_error( error, "reset GPU 0" );

	cudaSetDevice( gpu_1 );
	error = cudaFree( d_a[1] );
	process_error( error, "free memory on GPU 1" );
	error = cudaDeviceReset();
	process_error( error, "reset GPU 1" );
	
	printf("CUDA: %s\n", cudaGetErrorString( cudaGetLastError() ) );

	return 0;
}
