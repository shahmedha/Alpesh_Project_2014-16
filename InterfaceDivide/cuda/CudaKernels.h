#ifndef CUDAKERNELS_H_
#define CUDAKERNELS_H_

#include <map>
#include <set>
#include "cudaTypes.h"
#include "../sff/siena/types.h"
#include <iostream>
#include "../common/Timer.h"
#include "../common/Consts.h"

using namespace std;
using namespace siena;

class CudaKernels {
public:

	/**
	 * Constructor
	 */
	CudaKernels(int gpu_id);

	/**
	 * Destructor
	 */
	virtual ~CudaKernels();

	/**
	 * Config an interface by setting its predicate (the set of its filters)
	 */
	void ifConfig(int interfaceId, set<CudaFilter *> &filters);

	/**
	 * Call this after all interfaces have been set
	 */
	void consolidate();

	/**
	 * Processes the given messages, storing the set of matching interfaces into the results set
	 */
	void processMessage(CudaOutbox *outbox);

	/**
	 * Returns the statistics about processing time distribution
	 */
	void getStats(double &hToD, double &exec, double &dToH);

	/**
	* Returns number of Gpus available
	*/
	int getNumOfGpu();
	/**

	*/
	void getMatchingInterfaces(set<int> &results);

	void synco();

private:
    int gpu;
	map<int, set<CudaFilter *> > hostFilters;		// Interface -> set of filters (in host memory)
	int numFilters;						// Number of filters
	int numInterfaces;					// Number of interfaces
	map<string_t, int> numConstraints;			// Name -> number of constraints with that name
	map<string_t, Type> nameType;				// Name -> type of constraints with that name
	bool consolidated;					// True if the consolidate method has been called

	map<string_t, void *> nameDeviceConstrVal;		// Name -> set of constraint values (in device memory)
	map<string_t, Op *> nameDeviceConstrOp;			// Name -> set of constraint operands (in device memory)
	map<string_t, int *> nameDeviceFilterIdx;		// Name -> set of filter indexes (in device memory)
	unsigned char *currentFiltersCount;			// Filter -> number of constraints satisfied (in device memory)
	FilterInfo *filtersInfo;				// Filter -> information for that filter (in device memory)
	unsigned char *interfacesDevice;			// Pointer to the array of interfaces in device memory
	unsigned char *interfacesHost;				// Pointer to the array of interfaces in host memory

	CudaInputElem *hostInput;				// Pointer to the input (in host memory)
	CudaInputElem *deviceInput;				// Pointer to the input (in device memory)
	int numValues;						// Number of values in current input

	double hostToDeviceCopyTime;
	double execTime;
	double deviceToHostCopyTime;

	inline int copyMsgToDevice(CudaMessage *msg);
	inline void computeResults(int maxConstraints);
};

#endif /* CUDAKERNELS_H_ */
