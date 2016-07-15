#include "CudaKernelsSimpleBloom.h"

#define NUM_THREADS 256

static __constant__ CudaSBFInputElem constInput[MAX_ATTR_NUM];

static __device__ bool cuda_strcmp(char *s1, char *s2) {
	for ( ; *s1==*s2; ++s1, ++s2) {
		if (*s1=='\0') return true;
	}
	return false;
}

static __device__ bool cuda_prefix(char *s1, char *s2) {
	for ( ; *s1==*s2; ++s1, ++s2) {
		if (*(s2+1)=='\0') return true;
	}
	return false;
}

static __device__ bool cuda_substr(char *s1, char *s2) {
	int size1 = 0;
	int size2 = 0;
	while (s1[size1]!='\0') size1++;
	while (s2[size2]!='\0') size2++;
	if (size1==size2) return cuda_strcmp(s1, s2);
	if (size1<size2) return false;
	for (int i=0; i<size1-size2+1; i++) {
		bool failed = false;
		for (int j=0; j<size2; j++) {
			if (s1[i+j-1]!=s2[j]) {
				failed = true;
				break;
			}
		}
		if (! failed) return true;
	}
	return false;
}

static __global__ void cleanCounters(unsigned int *filtersCount, unsigned char *interfaces, const int numFilters, const int numInterfaces) {
	int pos = blockIdx.x*blockDim.x+threadIdx.x;
	// initialize interfaces and filtersCount
	if (pos<numInterfaces) interfaces[pos] = 0;
	while(pos<numFilters) {
		filtersCount[pos] = 0;
		pos = pos + gridDim.x*blockDim.x;
	}
}

template <int byteId>
static __global__ void evalConstraint(unsigned int *filtersCount, const FilterInfo *filterInfo, unsigned char *interfaces, const int numFilters, const int numInterfaces, const BFilter inputBF) {
	int constraintsIndex = blockIdx.x*blockDim.x+threadIdx.x;
	if (constraintsIndex>=constInput[blockIdx.y].numConstraints) return;
	CudaSBFInputElem inputElem = constInput[blockIdx.y];
	if((inputBF & inputElem.filterNames[constraintsIndex])!=inputElem.filterNames[constraintsIndex]) return;
	CudaValue val = inputElem.value;
	Op constrOp = inputElem.constrOp[constraintsIndex];
	if (val.type==INT) {
		IntCudaConstraint constrVal = ((IntCudaConstraint *)inputElem.constrVal)[constraintsIndex];
		if ((constrOp==EQ && val.intVal!=constrVal.value) ||
				(constrOp==LT && val.intVal>=constrVal.value) ||
				(constrOp==GT && val.intVal<=constrVal.value) ||
				(constrOp==DF && val.intVal==constrVal.value)) return;
	} else {
		StringCudaConstraint constrVal = ((StringCudaConstraint *)inputElem.constrVal)[constraintsIndex];
		if ((constrOp==EQ && !cuda_strcmp(val.stringVal, constrVal.value)) ||
				(constrOp==DF &&  cuda_strcmp(val.stringVal, constrVal.value)) ||
				(constrOp==PF && !cuda_prefix(val.stringVal, constrVal.value)) ||
				(constrOp==IN && !cuda_substr(val.stringVal, constrVal.value))) return;
	}
	int filterIndex = inputElem.filterIdx[constraintsIndex];
	int count;
	if (byteId==1) {
		count = atomicAdd(&filtersCount[filterIndex], 1);
	} else if (byteId==2) {
		count = atomicAdd(&filtersCount[filterIndex], 256);
		count = (count >> 8) & 0xF;
	} else if (byteId==3) {
		count = atomicAdd(&filtersCount[filterIndex], 65536);
		count = (count >> 16) & 0xF;
	} else if (byteId==4) {
		count = atomicAdd(&filtersCount[filterIndex], 16777216);
		count = (count >> 24) & 0xF;
	}
	if (count+1==filterInfo[filterIndex].numConstraints) {
		interfaces[filterInfo[filterIndex].interface] = 1;
	}
}

static inline void add(BFilter &bf1, const char* name) {
	bf1 = bf1 | 1u << ((name[0]+name[1])%(sizeof(BFilter)*8));
	bf1 = bf1 | 1u << ((name[2]+name[3])%(sizeof(BFilter)*8));
	bf1 = bf1 | 1u << ((name[4]+name[5])%(sizeof(BFilter)*8));
	bf1 = bf1 | 1u << ((name[6]+name[7])%(sizeof(BFilter)*8));
}

CudaKernelsSimpleBloom::CudaKernelsSimpleBloom() {
	currentByteId = 1;
	numInterfaces = 0;
	numFilters = 0;
	consolidated = false;
	hostToDeviceCopyTime = 0;
	execTime = 0;
	deviceToHostCopyTime = 0;
}

CudaKernelsSimpleBloom::~CudaKernelsSimpleBloom() {
	if (consolidated) {
		for (map<string_t, void *>::iterator it=nameDeviceConstrVal.begin(); it!=nameDeviceConstrVal.end(); ++it) {
			void *constrPtr = it->second;
			cudaFree(constrPtr);
		}
		for (map<string_t, Op *>::iterator it=nameDeviceConstrOp.begin(); it!=nameDeviceConstrOp.end(); ++it) {
			Op *constrPtr = it->second;
			cudaFree(constrPtr);
		}
		for (map<string_t, int *>::iterator it=nameDeviceFilterIdx.begin(); it!=nameDeviceFilterIdx.end(); ++it) {
			int *filterIdxPtr = it->second;
			cudaFree(filterIdxPtr);
		}
		for (map<string_t, BFilter *>::iterator it=nameDeviceFilterNames.begin(); it!=nameDeviceFilterNames.end(); ++it) {
			BFilter *filterNamePtr = it->second;
			cudaFree(filterNamePtr);
		}
		cudaFreeHost(hostInput);
		cudaFree(currentFiltersCount);
		cudaFree(filtersInfo);
		cudaFree(interfacesDevice);
		cudaFreeHost(interfacesHost);
	}
	for (map<int, set<CudaFilter *> >::iterator it=hostFilters.begin(); it!=hostFilters.end(); ++it) {
		for (set<CudaFilter *>::iterator it2=it->second.begin(); it2!=it->second.end(); ++it2) {
			CudaFilter *filter = *it2;
			delete filter;
		}
	}
}

void CudaKernelsSimpleBloom::ifConfig(int interfaceId, set<CudaFilter *> &filters) {
	// record the set of filters associated to this interface
	hostFilters.insert(make_pair(interfaceId, filters));

	// update the numConstraints and nameType data structures (to be used at consolidate time)
	for (set<CudaFilter *>::iterator it=filters.begin(); it!=filters.end(); ++it) {
		CudaFilter *filter = *it;
		for (int i=0; i<filter->numConstraints; i++) {
			string_t nameStr = filter->constraints[i].name;
			map<string_t, int>::iterator it=numConstraints.find(nameStr);
			if (it==numConstraints.end()) {
				numConstraints.insert(make_pair(nameStr, 1));
			} else {
				it->second++;
			}
			map<string_t, Type>::iterator it1=nameType.find(nameStr);
			if (it1==nameType.end()) {
				nameType.insert(make_pair(nameStr, filter->constraints[i].value.type));
			}
		}
		numFilters++;
	}
}

void CudaKernelsSimpleBloom::consolidate() {
	// allocate memory on device and host
	int e = 0;
	int allocSize = 0;
	numInterfaces = hostFilters.size();
	allocSize += sizeof(CudaSBFInputElem)*MAX_ATTR_NUM;  // allocated into constant memory (see static variable at the beginning of file)
	e += cudaMallocHost((void**) &hostInput, (size_t) sizeof(CudaSBFInputElem)*MAX_ATTR_NUM);
	e += cudaMalloc((void**) &interfacesDevice, (size_t) sizeof(unsigned char)*numInterfaces);
	allocSize += sizeof(unsigned char)*numInterfaces;
	e += cudaMallocHost((void**) &interfacesHost, (size_t) sizeof(unsigned char)*numInterfaces);
	map<string_t, int> currentNumConstraints;
	map<string_t, void *> nameHostConstrVal;
	map<string_t, Op *> nameHostConstrOp;
	map<string_t, int *> nameHostFilterIdx;
	map<string_t, BFilter *> nameHostFilterNames;
	for (map<string_t, int>::iterator it=numConstraints.begin(); it!=numConstraints.end(); ++it) {
		string_t name = it->first;
		int num = it->second;
		void *constrValPtr, *hostConstrValPtr;
		if(nameType[name]==INT) {
			e += cudaMalloc((void**) &constrValPtr, (size_t) sizeof(IntCudaConstraint)*num);
			hostConstrValPtr = malloc(sizeof(IntCudaConstraint)*num);
			allocSize += sizeof(IntCudaConstraint)*num;
		} else {
			e += cudaMalloc((void**) &constrValPtr, (size_t) sizeof(StringCudaConstraint)*num);
			hostConstrValPtr = malloc(sizeof(StringCudaConstraint)*num);
			allocSize += sizeof(StringCudaConstraint)*num;
		}
		nameDeviceConstrVal.insert(make_pair(name, constrValPtr));
		nameHostConstrVal.insert(make_pair(name, hostConstrValPtr));
		Op *constrOpPtr, *hostConstrOpPtr;
		e+= cudaMalloc((void**) &constrOpPtr, (size_t) sizeof(Op)*num);
		hostConstrOpPtr = (Op *)malloc(sizeof(Op)*num);
		allocSize += sizeof(Op)*num;
		nameDeviceConstrOp.insert(make_pair(name, constrOpPtr));
		nameHostConstrOp.insert(make_pair(name, hostConstrOpPtr));
		currentNumConstraints.insert(make_pair(name, 0));
		int *filterIdxPtr, *hostFilterIdxPtr;
		e+= cudaMalloc((void**) &filterIdxPtr, (size_t) sizeof(int)*num);
		hostFilterIdxPtr = (int *)malloc(sizeof(int)*num);
		allocSize += sizeof(int)*num;
		nameDeviceFilterIdx.insert(make_pair(name, filterIdxPtr));
		nameHostFilterIdx.insert(make_pair(name, hostFilterIdxPtr));
		BFilter *filterNamesPtr, *hostFilterNamesPtr;
		e+= cudaMalloc((void**) &filterNamesPtr, (size_t) sizeof(BFilter)*num);
		hostFilterNamesPtr = (BFilter *)malloc(sizeof(int)*num);
		allocSize += sizeof(BFilter)*num;
		nameDeviceFilterNames.insert(make_pair(name, filterNamesPtr));
		nameHostFilterNames.insert(make_pair(name, hostFilterNamesPtr));
	}
	e += cudaMalloc((void**) &currentFiltersCount, (size_t) sizeof(unsigned int)*numFilters);
	allocSize += sizeof(unsigned int)*numFilters;
	e += cudaMalloc((void**) &filtersInfo, (size_t) sizeof(FilterInfo)*numFilters);
	allocSize += sizeof(FilterInfo)*numFilters;
	if (e>0) {
		cerr << " Allocation error " << e << endl;
		exit(1);
	}

	// initialize the nameHostConstrVal, nameHostConstrOp, nameHostFilterIdx, nameHostFilterNames, and hostFiltersInfo structures
	// (to be copied into the corresponding structures in device later)
	int filterId = 0;
	FilterInfo *hostFiltersInfo = (FilterInfo *) malloc(sizeof(FilterInfo)*numFilters);
	for (map<int, set<CudaFilter *> >::iterator it=hostFilters.begin(); it!=hostFilters.end(); ++it) {
		int interfaceId = it->first;
		for (set<CudaFilter *>::iterator it2=it->second.begin(); it2!=it->second.end(); ++it2) {
			CudaFilter *filter = *it2;
			BFilter bf=0;
			for (int i=0; i<filter->numConstraints; i++) {
				add(bf,filter->constraints[i].name);
			}
			for (int i=0; i<filter->numConstraints; i++) {
				string_t name = filter->constraints[i].name;
				int writingIndex = currentNumConstraints[name];
				currentNumConstraints[name] = writingIndex+1;
				Op *hostConstrOpPtr = nameHostConstrOp[name];
				hostConstrOpPtr[writingIndex] = filter->constraints[i].op;
				if(nameType[name]==INT) {
					IntCudaConstraint *hostConstrValPtr = (IntCudaConstraint *)nameHostConstrVal[name];
					hostConstrValPtr[writingIndex].value = filter->constraints[i].value.intVal;
				} else {
					StringCudaConstraint *hostConstrValPtr = (StringCudaConstraint *)nameHostConstrVal[name];
					memcpy(hostConstrValPtr[writingIndex].value, filter->constraints[i].value.stringVal, STRING_VAL_LEN);
				}
				int *hostFilterIdxPtr = nameHostFilterIdx[name];
				hostFilterIdxPtr[writingIndex] = filterId;
				BFilter *hostFilterNamesPtr = nameHostFilterNames[name];
				hostFilterNamesPtr[writingIndex] = bf;
			}
			hostFiltersInfo[filterId].numConstraints = filter->numConstraints;
			hostFiltersInfo[filterId].interface = interfaceId;
			filterId++;
		}
	}

	// initialize the device memory
	for (map<string_t, void *>::iterator it=nameHostConstrVal.begin(); it!=nameHostConstrVal.end(); ++it) {
		string_t name = it->first;
		void *host = it->second;
		void *device = nameDeviceConstrVal[name];
		int size = numConstraints[name];
		if(nameType[name]==INT) {
			e += cudaMemcpy(device, host, sizeof(IntCudaConstraint)*size, cudaMemcpyHostToDevice);
		} else {
			e += cudaMemcpy(device, host, sizeof(StringCudaConstraint)*size, cudaMemcpyHostToDevice);
		}
		cudaDeviceSynchronize();
		free(host);
	}
	for (map<string_t, Op *>::iterator it=nameHostConstrOp.begin(); it!=nameHostConstrOp.end(); ++it) {
		string_t name = it->first;
		Op *host = it->second;
		Op *device = nameDeviceConstrOp[name];
		int size = numConstraints[name];
		e += cudaMemcpy(device, host, sizeof(Op)*size, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		free(host);
	}
	for (map<string_t, int *>::iterator it=nameHostFilterIdx.begin(); it!=nameHostFilterIdx.end(); ++it) {
		string_t name = it->first;
		int *host = it->second;
		int *device = nameDeviceFilterIdx[name];
		int size = numConstraints[name];
		e += cudaMemcpy(device, host, sizeof(int)*size, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		free(host);
	}
	for (map<string_t, BFilter *>::iterator it=nameHostFilterNames.begin(); it!=nameHostFilterNames.end(); ++it) {
		string_t name = it->first;
		BFilter *host = it->second;
		BFilter *device = nameDeviceFilterNames[name];
		int size = numConstraints[name];
		e += cudaMemcpy(device, host, sizeof(BFilter)*size, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		free(host);
	}
	e += cudaMemcpy(filtersInfo, hostFiltersInfo, (size_t) sizeof(FilterInfo)*numFilters, cudaMemcpyHostToDevice);
	cudaMemset(currentFiltersCount, 0, (size_t) sizeof(unsigned int)*numFilters);
	cudaMemset(interfacesDevice, 0, (size_t) sizeof(unsigned char)*numInterfaces);
	cudaDeviceSynchronize();
	consolidated = true;
	if (e>0) {
		cerr << " Memcpy error " << e << " during consolidation " <<  endl;
		exit(1);
	}
	free(hostFiltersInfo);

	// set up the runtime to optimize performance
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	int totConstr=0;
	for(map<string_t,int>::iterator it=numConstraints.begin(); it!=numConstraints.end(); ++it) {
		totConstr+=it->second;
	}
	cout << endl << " ### " << totConstr << " constraints allocated ### " << endl;
	cout << endl << " ### " << allocSize << " bytes allocated on device ### " << endl;
	cout << endl << "#####################" << endl;
}

void CudaKernelsSimpleBloom::getStats(double &hToD, double &exec, double &dToH) {
	hToD = hostToDeviceCopyTime;
	exec = execTime;
	dToH = deviceToHostCopyTime;
}

#if STATS==1
void CudaKernelsSimpleBloom::processMessage(CudaOutbox *outbox) {
	Timer t;
	t.start();
	BFilter messageNames = copyMsgToDevice(outbox->message);
	//cudaDeviceSynchronize(); // TODO: remove
	hostToDeviceCopyTime += t.stop();
	t.start();
	computeResults(messageNames);
	//cudaDeviceSynchronize(); // TODO: remove
	execTime += t.stop();
	t.start();
	getMatchingInterfaces(outbox->outgoingInterfaces);
	//cudaDeviceSynchronize(); // TODO: remove
	deviceToHostCopyTime += t.stop();
}
#elif STATS==0
void CudaKernelsSimpleBloom::processMessage(CudaOutbox *outbox) {
	BFilter messageNames = copyMsgToDevice(outbox->message);
	computeResults(messageNames);
	getMatchingInterfaces(outbox->outgoingInterfaces);
}
#endif

BFilter CudaKernelsSimpleBloom::copyMsgToDevice(CudaMessage *message) {
	BFilter bf=0;
	int dest = 0;
	for (int i=0; i<message->numAttributes; i++) {
		add(bf, message->attributes[i].name);
		string_t name = message->attributes[i].name;
		hostInput[dest].constrVal = nameDeviceConstrVal[name];
		hostInput[dest].constrOp = nameDeviceConstrOp[name];
		hostInput[dest].filterIdx = nameDeviceFilterIdx[name];
		hostInput[dest].filterNames = nameDeviceFilterNames[name];
		hostInput[dest].numConstraints = numConstraints[name];
		hostInput[dest].value = message->attributes[i].value;
		dest++;
	}
	numValues = dest;
	if (dest>0) {
		int e = 0;
		e += cudaMemcpyToSymbolAsync(constInput, hostInput, (size_t) sizeof(CudaSBFInputElem)*numValues);
		if (e>0) {
			cerr << " Memcpy error " << e << " during message processing " <<  endl;
			exit(1);
		}
	}
	return bf;
}

void CudaKernelsSimpleBloom::computeResults(BFilter messageNames) {
	int maxNumConstr = 0;
	for(int i=0; i<numValues; i++) {
		if(maxNumConstr<hostInput[i].numConstraints) maxNumConstr=hostInput[i].numConstraints;
	}
	switch (currentByteId) {
	case 1:
		evalConstraint<1><<<dim3(maxNumConstr/NUM_THREADS+1,numValues), NUM_THREADS>>>(currentFiltersCount, filtersInfo, interfacesDevice, numFilters, numInterfaces, messageNames);
		break;
	case 2:
		evalConstraint<2><<<dim3(maxNumConstr/NUM_THREADS+1,numValues), NUM_THREADS>>>(currentFiltersCount, filtersInfo, interfacesDevice, numFilters, numInterfaces, messageNames);
		break;
	case 3:
		evalConstraint<3><<<dim3(maxNumConstr/NUM_THREADS+1,numValues), NUM_THREADS>>>(currentFiltersCount, filtersInfo, interfacesDevice, numFilters, numInterfaces, messageNames);
		break;
	case 4:
		evalConstraint<4><<<dim3(maxNumConstr/NUM_THREADS+1,numValues), NUM_THREADS>>>(currentFiltersCount, filtersInfo, interfacesDevice, numFilters, numInterfaces, messageNames);
		break;
	}
}

void CudaKernelsSimpleBloom::getMatchingInterfaces(set<int> &results) {
	int e = cudaMemcpyAsync(interfacesHost, interfacesDevice, (size_t) sizeof(unsigned char)*numInterfaces, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	//cudaMemsetAsync(currentFiltersCount, 0, (size_t) sizeof(unsigned int)*numFilters);
	if (currentByteId!=4) {
		cudaMemsetAsync(interfacesDevice, 0, (size_t) sizeof(unsigned char)*numInterfaces);
	} else {
		cleanCounters<<<numFilters/2048, NUM_THREADS>>>(currentFiltersCount, interfacesDevice, numFilters, numInterfaces);
	}
	currentByteId++;
	if (currentByteId>4) currentByteId = 1;
	if (e>0) {
		cerr << " Memcpy error " << e << " while copying matching interfaces " <<  endl;
		exit(1);
	}
	for (int i=0; i<numInterfaces; i++) {
		if (interfacesHost[i]!=0) {
			results.insert(i);
		}
	}
}
