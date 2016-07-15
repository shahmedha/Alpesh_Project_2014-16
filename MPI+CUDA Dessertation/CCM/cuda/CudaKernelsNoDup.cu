#include "CudaKernelsNoDup.h"

#define NUM_THREADS 256

static __constant__ CudaInputElem constInput[MAX_ATTR_NUM];

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

static __global__ void cleanCounters(unsigned char *filtersCount, unsigned char *interfaces, const int numFilters, const int numInterfaces) {
  int pos = blockIdx.x*blockDim.x+threadIdx.x;
  // initialize interfaces and filtersCount
  if (pos<numInterfaces) interfaces[pos] = 0;
  while(pos<numFilters) {
    filtersCount[pos] = 0;
    pos = pos + gridDim.x*blockDim.x;
  }
}

static __global__ void evalConstraint(unsigned char *filtersCount, const FilterInfo *filterInfo, unsigned char *interfaces, const int numFilters, const int numInterfaces, int attributeIdx) {
  int constraintsIndex = blockIdx.x*blockDim.x+threadIdx.x;
  if (constraintsIndex>=constInput[attributeIdx].numConstraints) return;
  CudaInputElem inputElem = constInput[attributeIdx];
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
  int *filterIndex = inputElem.filterIdx+constraintsIndex*MAX_FILTERS_X_CONSTR;
  for(int i=0; i<MAX_FILTERS_X_CONSTR; i++) {
  filtersCount[filterIndex[i]]++;
  }
  // TODO: Should use the following to avoid errors. Previous one overestimate filterId=0
  /*
  for(int i=0; i<MAX_FILTERS_X_CONSTR; i++) {
    if(filterIndex[i]!=-1) filtersCount[filterIndex[i]]++;
    else break;
  }
  */
}

static __global__ void summarize(unsigned char *filtersCount, const FilterInfo *filterInfo, unsigned char *interfaces, const int numFilters, const int numInterfaces) {
  int pos = blockIdx.x*blockDim.x+threadIdx.x;
  while(pos<numFilters) {
    if (filtersCount[pos]==filterInfo[pos].numConstraints) {
      interfaces[filterInfo[pos].interface] = 1;
    }
    pos = pos + gridDim.x*blockDim.x;
  }
}

CudaKernelsNoDup::CudaKernelsNoDup() {
  numInterfaces = 0;
  numFilters = 0;
  consolidated = false;
  hostToDeviceCopyTime = 0;
  execTime = 0;
  deviceToHostCopyTime = 0;
}

CudaKernelsNoDup::~CudaKernelsNoDup() {
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

void CudaKernelsNoDup::ifConfig(int interfaceId, set<CudaFilter *> &filters) {
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

void CudaKernelsNoDup::consolidate() {
  // allocate memory on device and host
  int e = 0;
  int allocSize = 0;
  numInterfaces = hostFilters.size();
  allocSize += sizeof(CudaInputElem)*MAX_ATTR_NUM;  // allocated into constant memory (see static variable at the beginning of file)
  e += cudaMallocHost((void**) &hostInput, (size_t) sizeof(CudaInputElem)*MAX_ATTR_NUM);
  e += cudaMalloc((void**) &interfacesDevice, (size_t) sizeof(unsigned char)*numInterfaces);
  allocSize += sizeof(unsigned char)*numInterfaces;
  e += cudaMallocHost((void**) &interfacesHost, (size_t) sizeof(unsigned char)*numInterfaces);
  map<string_t, int> currentNumConstraints;
  map<string_t, void *> nameHostConstrVal;
  map<string_t, Op *> nameHostConstrOp;
  map<string_t, int *> nameHostFilterIdx;
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
    e+= cudaMalloc((void**) &filterIdxPtr, (size_t) sizeof(int)*num*MAX_FILTERS_X_CONSTR);
    hostFilterIdxPtr = (int *)malloc(sizeof(int)*num*MAX_FILTERS_X_CONSTR);
    allocSize += sizeof(int)*num*MAX_FILTERS_X_CONSTR;
    nameDeviceFilterIdx.insert(make_pair(name, filterIdxPtr));
    nameHostFilterIdx.insert(make_pair(name, hostFilterIdxPtr));
    currentNumConstraints.insert(make_pair(name, 0));
  }
  e += cudaMalloc((void**) &currentFiltersCount, (size_t) sizeof(unsigned char)*numFilters);
  allocSize += sizeof(unsigned char)*numFilters;
  e += cudaMalloc((void**) &filtersInfo, (size_t) sizeof(FilterInfo)*numFilters);
  allocSize += sizeof(FilterInfo)*numFilters;
  if (e>0) {
    cerr << " Allocation error " << e << endl;
    exit(1);
  }

  // initialize the nameHostConstrVal, nameHostConstrOp, nameHostFilterIdx, and hostFiltersInfo structures (to be copied into the corresponding structures in device later)
  int filterId = 0;
  FilterInfo *hostFiltersInfo = (FilterInfo *) malloc(sizeof(FilterInfo)*numFilters);
  for (map<int, set<CudaFilter *> >::iterator it=hostFilters.begin(); it!=hostFilters.end(); ++it) {
    int interfaceId = it->first;
    for (set<CudaFilter *>::iterator it2=it->second.begin(); it2!=it->second.end(); ++it2) {
      CudaFilter *filter = *it2;
      for (int i=0; i<filter->numConstraints; i++) {
	string_t name = filter->constraints[i].name;
	int writingIndex = currentNumConstraints[name];
	// search if constraint is already present
	bool found=false, goon=true;
	for(int k=writingIndex-1; k>=0 && goon; k--) {
	  Op *hostConstrOpPtr = nameHostConstrOp[name];
	  if(hostConstrOpPtr[k]!=filter->constraints[i].op) continue;
	  if(nameType[name]==INT) {
	    IntCudaConstraint *hostConstrValPtr = (IntCudaConstraint *)nameHostConstrVal[name];
	    if(hostConstrValPtr[k].value != filter->constraints[i].value.intVal) continue;
	  } else {
	    StringCudaConstraint *hostConstrValPtr = (StringCudaConstraint *)nameHostConstrVal[name];
	    if(strncmp(hostConstrValPtr[k].value,filter->constraints[i].value.stringVal,STRING_VAL_LEN)!=0) continue;
	  }
	  goon=false; // no need to continue searching for the same constraint as I found it
	  int *hostFilterIdxPtr = nameHostFilterIdx[name];
	  for(int j=0; j<MAX_FILTERS_X_CONSTR; j++) {
	    if(hostFilterIdxPtr[k*MAX_FILTERS_X_CONSTR+j]==0) {  // TODO: ==-1
	      hostFilterIdxPtr[k*MAX_FILTERS_X_CONSTR+j] = filterId;
	      found=true;
	      break;
	    }
	  }
	}
	if(found) continue; // go to the next constraint
	// constraint is not present, add it at the end
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
	hostFilterIdxPtr[writingIndex*MAX_FILTERS_X_CONSTR] = filterId;
	for(int j=1; j<MAX_FILTERS_X_CONSTR; j++) hostFilterIdxPtr[writingIndex*MAX_FILTERS_X_CONSTR+j] = 0; // TODO: =-1
      }
      hostFiltersInfo[filterId].numConstraints = filter->numConstraints;
      hostFiltersInfo[filterId].interface = interfaceId;
      filterId++;
    }
  }

  // update numConstraints according to the real number of non dup constraints
  for(map<string_t,int>::iterator it=currentNumConstraints.begin(); it!=currentNumConstraints.end(); ++it) {
    numConstraints[it->first] = it->second;
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
    e += cudaMemcpy(device, host, sizeof(int)*size*MAX_FILTERS_X_CONSTR, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    free(host);
  }
  e += cudaMemcpy(filtersInfo, hostFiltersInfo, (size_t) sizeof(FilterInfo)*numFilters, cudaMemcpyHostToDevice);
  cudaMemset(currentFiltersCount, 0, (size_t) sizeof(unsigned char)*numFilters);
  cudaMemset(interfacesDevice, 0, (size_t) sizeof(unsigned char)*numInterfaces);
  cudaDeviceSynchronize();
  consolidated = true;
  if (e>0) {
    cerr << " Memcpy error " << e << " during consolidation " <<  endl;
    exit(1);
  }
  free(hostFiltersInfo);
  
  // set up the runtime to optimize performance
  //cudaFuncSetCacheConfig(evalConstraint, cudaFuncCachePreferL1);
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  int totConstr=0;
  for(map<string_t,int>::iterator it=numConstraints.begin(); it!=numConstraints.end(); ++it) {
    totConstr+=it->second;
  }
  cout << endl << " ### " << totConstr << " constraints allocated ### " << endl;
  cout << endl << " ### " << allocSize << " bytes allocated on device ### " << endl;
  cout << endl << "#####################" << endl;
}

void CudaKernelsNoDup::getStats(double &hToD, double &exec, double &dToH) {
  hToD = hostToDeviceCopyTime;
  exec = execTime;
  dToH = deviceToHostCopyTime;
}

#if STATS==1
void CudaKernelsNoDup::processMessage(CudaOutbox *outbox) {
  Timer t;
  t.start();
  int maxConstr = copyMsgToDevice(outbox->message);
  hostToDeviceCopyTime += t.stop();
  if (maxConstr>0) {
    t.start();
    computeResults(maxConstr);
    execTime += t.stop();
    t.start();
    getMatchingInterfaces(outbox->outgoingInterfaces);
    deviceToHostCopyTime += t.stop();
  }
}
#elif STATS==0
void CudaKernelsNoDup::processMessage(CudaOutbox *outbox) {
  int maxConstr = copyMsgToDevice(outbox->message);
  if (maxConstr>0) {
    computeResults(maxConstr);
    getMatchingInterfaces(outbox->outgoingInterfaces);
  }
}
#endif

int CudaKernelsNoDup::copyMsgToDevice(CudaMessage *message) {
  int dest = 0;
  int maxConstr = 0;
  for (int i=0; i<message->numAttributes; i++) {
    string_t name = message->attributes[i].name;
    map<string_t, void *>::iterator it = nameDeviceConstrVal.find(name);
    if(it==nameDeviceConstrVal.end()) {
      cerr << "Name: "; 
      for(int i=0; i<name.length(); i++) cerr << name[i]; 
      cerr << " not found during message processing" << endl;
      exit(1);
    }
    hostInput[dest].constrVal = it->second;
    map<string_t, Op *>::iterator it1 = nameDeviceConstrOp.find(name);
    if(it1==nameDeviceConstrOp.end()) {
      cerr << "Name: "; 
      for(int i=0; i<name.length(); i++) cerr << name[i]; 
      cerr << " not found during message processing" << endl;
      exit(1);
    }
    hostInput[dest].constrOp = it1->second;
    map<string_t, int *>::iterator it2 = nameDeviceFilterIdx.find(name);
    if(it2==nameDeviceFilterIdx.end()) {
      cerr << "Name: "; 
      for(int i=0; i<name.length(); i++) cerr << name[i]; 
      cerr << " not found during message processing" << endl;
      exit(1);
    }
    hostInput[dest].filterIdx = it2->second;
    hostInput[dest].numConstraints = numConstraints[name];
    if (hostInput[dest].numConstraints>maxConstr) maxConstr = hostInput[dest].numConstraints;
    hostInput[dest].value = message->attributes[i].value;
    dest++;
  }
  numValues = dest;
  if (dest>0) {
    int e = 0;
    e += cudaMemcpyToSymbolAsync(constInput, hostInput, (size_t) sizeof(CudaInputElem)*numValues);
    if (e>0) {
      cerr << " Memcpy error " << e << " during message processing " <<  endl;
      exit(1);
    }
  }
  return maxConstr;
}

void CudaKernelsNoDup::computeResults(int maxConstr) {
	//int numBlocksX = 1+maxConstr/NUM_THREADS;
	//dim3 numBlocks = dim3(numBlocksX);
	for(int i=0; i<numValues; i++) {
	  evalConstraint<<<hostInput[i].numConstraints/NUM_THREADS+1, NUM_THREADS>>>(currentFiltersCount, filtersInfo, interfacesDevice, numFilters, numInterfaces, i);
	}
	summarize<<<numFilters/2048, NUM_THREADS>>>(currentFiltersCount, filtersInfo, interfacesDevice, numFilters, numInterfaces);
}

void CudaKernelsNoDup::getMatchingInterfaces(set<int> &results) {
	int e = cudaMemcpyAsync(interfacesHost, interfacesDevice, (size_t) sizeof(unsigned char)*numInterfaces, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaMemsetAsync(currentFiltersCount, 0, (size_t) sizeof(unsigned char)*numFilters);
	cudaMemsetAsync(interfacesDevice, 0, (size_t) sizeof(unsigned char)*numInterfaces);
	//cleanCounters<<<numFilters/2048, NUM_THREADS>>>(currentFiltersCount, interfacesDevice, numFilters, numInterfaces);
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
