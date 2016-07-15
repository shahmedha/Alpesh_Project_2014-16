#ifndef CUDAKERNELSBLOOM_H_
#define CUDAKERNELSBLOOM_H_

#include <map>
#include <set>
#include "cudaTypes.h"
#include "../sff/siena/types.h"
#include <iostream>
#include "../common/Timer.h"
#include "../common/Consts.h"

using namespace std;
using namespace siena;

class CudaKernelsBloom {
public:

	/**
	 * Constructor
	 */
	CudaKernelsBloom();

	/**
	 * Destructor
	 */
	virtual ~CudaKernelsBloom();

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

private:
	map<int, set<CudaFilter *> > hostFilters;		// Interface -> set of filters (in host memory)
	int numFilters;						// Number of filters
	int numInterfaces;					// Number of interfaces
	bool consolidated;					// True if the consolidate method has been called

	map<string_t, Type> nameType;				// Name -> type of constraints with that name
        map<string_t, int> numConstraints;                      // Name -> number of constraints with that name
	map<string_t, BFConstrList> nameDeviceBFConstr;		// Name -> list of Bloom constraints (in device memory)

	unsigned int *currentFiltersCount;			// Filter -> number of constraints satisfied (in device memory)
	FilterInfo *filtersInfo;				// Filter -> information for that filter (in device memory)
	unsigned char *interfacesDevice;			// Pointer to the array of interfaces in device memory
	unsigned char *interfacesHost;				// Pointer to the array of interfaces in host memory

	CudaBFInputElem *hostInput;				// Pointer to the input (in host memory)
	int numValues;						// Number of values in current input

	double hostToDeviceCopyTime;
	double execTime;
	double deviceToHostCopyTime;

	inline BFilter copyMsgToDevice(CudaMessage *msg);
	inline void computeResults(BFilter messageNames);
	inline void getMatchingInterfaces(set<int> &results);
};

#endif /* CUDAKERNELSBLOOM_H_ */
