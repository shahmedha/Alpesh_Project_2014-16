#include "CudaKernelsBloom.h"

#define NUM_THREADS 256

static __constant__ CudaBFInputElem constInput[MAX_ATTR_NUM];

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

static __global__ void evalConstraint(unsigned int *filtersCount, const FilterInfo *filterInfo, unsigned char *interfaces, const int numFilters, const int numInterfaces, const BFilter inputBF) {
  if(blockIdx.y>=constInput[blockIdx.z].numBFConstr) return;
  CudaBFConstr bfConstr = constInput[blockIdx.z].bfConstr[blockIdx.y];
  if((inputBF & bfConstr.filterNames)!=bfConstr.filterNames) return;
  int constraintsIndex = blockIdx.x*blockDim.x+threadIdx.x;
  if (constraintsIndex>=bfConstr.numConstr) return;
  CudaValue val = constInput[blockIdx.z].value;
  Op constrOp = bfConstr.constrOp[constraintsIndex];
  if (val.type==INT) {
    IntCudaConstraint constrVal = ((IntCudaConstraint *)bfConstr.constrVal)[constraintsIndex];
    if ((constrOp==EQ && val.intVal!=constrVal.value) ||
	(constrOp==LT && val.intVal>=constrVal.value) ||
	(constrOp==GT && val.intVal<=constrVal.value) ||
	(constrOp==DF && val.intVal==constrVal.value)) return;
  } else {
    StringCudaConstraint constrVal = ((StringCudaConstraint *)bfConstr.constrVal)[constraintsIndex];
    if ((constrOp==EQ && !cuda_strcmp(val.stringVal, constrVal.value)) ||
	(constrOp==DF &&  cuda_strcmp(val.stringVal, constrVal.value)) ||
	(constrOp==PF && !cuda_prefix(val.stringVal, constrVal.value)) ||
	(constrOp==IN && !cuda_substr(val.stringVal, constrVal.value))) return;
  }
  int filterIndex = bfConstr.filterIdx[constraintsIndex];
  int count = atomicAdd(&filtersCount[filterIndex], 1);
  if (count+1==filterInfo[filterIndex].numConstraints) {
    interfaces[filterInfo[filterIndex].interface] = 1;
  }
}

static inline void add(BFilter &bf1, const char* name) {
  bf1 = bf1 | 1u << ((name[0]+name[1]+name[2]+name[3])%(sizeof(BFilter)*8));
}

CudaKernelsBloom::CudaKernelsBloom() {
  numInterfaces = 0;
  numFilters = 0;
  consolidated = false;
  hostToDeviceCopyTime = 0;
  execTime = 0;
  deviceToHostCopyTime = 0;
}

CudaKernelsBloom::~CudaKernelsBloom() {
  if (consolidated) {
    for (map<string_t, BFConstrList>::iterator it=nameDeviceBFConstr.begin(); it!=nameDeviceBFConstr.end(); ++it) {
      string_t name = it->first;
      BFConstrList deviceBFCList = it->second;
      for(int i=0; i<deviceBFCList.numBFConstr; i++) {
	cudaFree(deviceBFCList.bfConstr[i].constrOp);
	cudaFree(deviceBFCList.bfConstr[i].constrVal);
	cudaFree(deviceBFCList.bfConstr[i].filterIdx);
      }
      free(deviceBFCList.bfConstr);
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

void CudaKernelsBloom::ifConfig(int interfaceId, set<CudaFilter *> &filters) {
  // record the set of filters associated to this interface
  hostFilters.insert(make_pair(interfaceId, filters));

  // calculate numFilters and build the nameType data structure (to be used at consolidate time)
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

void CudaKernelsBloom::consolidate() {
  // allocate memory on device and host and build the data structures required at run time
  int e = 0;
  int allocSize = 0;

  // initialize numInterfaces and build the hostInput, interfacesDevice and interfaceHost data structures
  numInterfaces = hostFilters.size();
  allocSize += sizeof(CudaBFInputElem)*MAX_ATTR_NUM;   // allocated into constant memory (see static variable at the beginning of file)
  e += cudaMallocHost((void**) &hostInput, (size_t) sizeof(CudaBFInputElem)*MAX_ATTR_NUM);
  e += cudaMalloc((void**) &interfacesDevice, (size_t) sizeof(unsigned char)*numInterfaces);
  allocSize += sizeof(unsigned char)*numInterfaces;
  e += cudaMallocHost((void**) &interfacesHost, (size_t) sizeof(unsigned char)*numInterfaces);

  // initialize the nameHostBFConstr and hostFiltersInfo structures (to be copied into the corresponding structures in device later)
  map<string_t, BFConstrList> nameHostBFConstr;
  int filterId = 0;
  FilterInfo *hostFiltersInfo = (FilterInfo *) malloc(sizeof(FilterInfo)*numFilters);
  for (map<int, set<CudaFilter *> >::iterator it=hostFilters.begin(); it!=hostFilters.end(); ++it) {
    int interfaceId = it->first;
    for (set<CudaFilter *>::iterator it2=it->second.begin(); it2!=it->second.end(); ++it2) {
      CudaFilter *filter = *it2;
      for (int i=0; i<filter->numConstraints; i++) {
	string_t name = filter->constraints[i].name;
	if(nameHostBFConstr.find(name)==nameHostBFConstr.end()) {
	  BFConstrList c;
	  c.bfConstr = (CudaBFConstr *) malloc(sizeof(CudaBFConstr)*NUM_DIFF_BF_IN_BFCONSTR);  // overallocating
	  memset(c.bfConstr, 0, sizeof(CudaBFConstr)*NUM_DIFF_BF_IN_BFCONSTR);
	  c.numBFConstr = 0;
	  c.maxNumConstr = 0;
	  nameHostBFConstr.insert(make_pair(name, c));
	}

	// build the bloom filter
	BFilter bf;
	bf=0;
	int k=0;
	for(int j=0; j<filter->numConstraints && k<MAX_NUM_NAMES_IN_BF; j++) {
	  if(j==i) continue;
	  add(bf, filter->constraints[j].name);
	  k++;
	}
	// find if this bloom filter is already present otherwise add at the end
	int pos=-1;
	for(int j=0; j<nameHostBFConstr[name].numBFConstr; j++) {
	  if(nameHostBFConstr[name].bfConstr[j].filterNames == bf) {pos=j; break;}
	}
	if(pos==-1) {
	  pos=nameHostBFConstr[name].numBFConstr;
	  nameHostBFConstr[name].numBFConstr++;
	  nameHostBFConstr[name].bfConstr[pos].filterNames = bf;
	  nameHostBFConstr[name].bfConstr[pos].numConstr = 0;
	  nameHostBFConstr[name].bfConstr[pos].constrOp = (Op *)malloc(sizeof(Op)*numConstraints[name]);  // overallocating
	  if(nameType[name]==INT) {
	    nameHostBFConstr[name].bfConstr[pos].constrVal = malloc(sizeof(IntCudaConstraint)*numConstraints[name]);  // overallocating
	  } else {
	    nameHostBFConstr[name].bfConstr[pos].constrVal = malloc(sizeof(StringCudaConstraint)*numConstraints[name]);  // overallocating
	  }
	  nameHostBFConstr[name].bfConstr[pos].filterIdx = (int *)malloc(sizeof(int)*numConstraints[name]);  // overallocating
	}
	// add this constraint into nameHostBFConstr[name].bfConstr[pos]
	nameHostBFConstr[name].bfConstr[pos].constrOp[nameHostBFConstr[name].bfConstr[pos].numConstr] = filter->constraints[i].op;
	if(nameType[name]==INT) {
	  IntCudaConstraint *val = (IntCudaConstraint *)nameHostBFConstr[name].bfConstr[pos].constrVal;
	  val[nameHostBFConstr[name].bfConstr[pos].numConstr].value = filter->constraints[i].value.intVal;
	} else {
	  StringCudaConstraint *val = (StringCudaConstraint *)nameHostBFConstr[name].bfConstr[pos].constrVal;
	  memcpy(val[nameHostBFConstr[name].bfConstr[pos].numConstr].value, filter->constraints[i].value.stringVal, STRING_VAL_LEN);
	}
	nameHostBFConstr[name].bfConstr[pos].filterIdx[nameHostBFConstr[name].bfConstr[pos].numConstr] = filterId;
	nameHostBFConstr[name].bfConstr[pos].numConstr++;
	if(nameHostBFConstr[name].maxNumConstr<nameHostBFConstr[name].bfConstr[pos].numConstr) nameHostBFConstr[name].maxNumConstr=nameHostBFConstr[name].bfConstr[pos].numConstr;
      }
      hostFiltersInfo[filterId].numConstraints = filter->numConstraints;
      hostFiltersInfo[filterId].interface = interfaceId;
      filterId++;
    }
  }

  // allocate and initialize the device memory building the nameDeviceBFConstr data structure
  Timer t;
  for (map<string_t, BFConstrList>::iterator it=nameHostBFConstr.begin(); it!=nameHostBFConstr.end(); ++it) {
    t.start();
    string_t name = it->first;
    BFConstrList hostBFCList = it->second;
    nameDeviceBFConstr[name].numBFConstr = hostBFCList.numBFConstr;
    nameDeviceBFConstr[name].maxNumConstr = hostBFCList.maxNumConstr;
    nameDeviceBFConstr[name].bfConstr = (CudaBFConstr *) malloc(sizeof(CudaBFConstr)*hostBFCList.numBFConstr);
    for(int i=0; i<hostBFCList.numBFConstr; i++) {
      int numConstr = nameHostBFConstr[name].bfConstr[i].numConstr;
      nameDeviceBFConstr[name].bfConstr[i].filterNames = hostBFCList.bfConstr[i].filterNames;
      nameDeviceBFConstr[name].bfConstr[i].numConstr = numConstr;
      e+=cudaMalloc(&nameDeviceBFConstr[name].bfConstr[i].constrOp, sizeof(Op)*numConstr);
      allocSize += sizeof(Op)*numConstr;
      e += cudaMemcpy(nameDeviceBFConstr[name].bfConstr[i].constrOp, hostBFCList.bfConstr[i].constrOp, sizeof(Op)*numConstr, cudaMemcpyHostToDevice);
      cudaDeviceSynchronize();
      if(nameType[name]==INT) {
	e+=cudaMalloc(&nameDeviceBFConstr[name].bfConstr[i].constrVal, sizeof(IntCudaConstraint)*numConstr);
	allocSize += sizeof(IntCudaConstraint)*numConstr;
	e += cudaMemcpy(nameDeviceBFConstr[name].bfConstr[i].constrVal, hostBFCList.bfConstr[i].constrVal, sizeof(IntCudaConstraint)*numConstr, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
      } else {
	e+=cudaMalloc(&nameDeviceBFConstr[name].bfConstr[i].constrVal, sizeof(StringCudaConstraint)*numConstr);
	allocSize += sizeof(StringCudaConstraint)*numConstr;
	e += cudaMemcpy(nameDeviceBFConstr[name].bfConstr[i].constrVal, hostBFCList.bfConstr[i].constrVal, sizeof(StringCudaConstraint)*numConstr, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
      }
      e+=cudaMalloc(&nameDeviceBFConstr[name].bfConstr[i].filterIdx, sizeof(int)*numConstr);
      allocSize += sizeof(int)*numConstr;
      e += cudaMemcpy(nameDeviceBFConstr[name].bfConstr[i].filterIdx, hostBFCList.bfConstr[i].filterIdx, sizeof(int)*numConstr, cudaMemcpyHostToDevice);
      cudaDeviceSynchronize();
      free(hostBFCList.bfConstr[i].constrOp);
      free(hostBFCList.bfConstr[i].constrVal);
      free(hostBFCList.bfConstr[i].filterIdx);
    }
    free(hostBFCList.bfConstr);
  }
  e += cudaMalloc(&currentFiltersCount, sizeof(unsigned int)*numFilters);
  allocSize += sizeof(unsigned int)*numFilters;
  e+=cudaMalloc(&filtersInfo, sizeof(FilterInfo)*numFilters);
  allocSize += sizeof(FilterInfo)*numFilters;
  e += cudaMemcpy(filtersInfo, hostFiltersInfo, (size_t) sizeof(FilterInfo)*numFilters, cudaMemcpyHostToDevice);
  free(hostFiltersInfo);
  cudaMemset(currentFiltersCount, 0, (size_t) sizeof(unsigned int)*numFilters);
  cudaMemset(interfacesDevice, 0, (size_t) sizeof(unsigned int)*numInterfaces);
  cudaDeviceSynchronize();
  consolidated = true;
  if (e>0) {
    cerr << " Memcpy error " << e << " during consolidation " <<  endl;
    exit(1);
  }
  
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

void CudaKernelsBloom::getStats(double &hToD, double &exec, double &dToH) {
  hToD = hostToDeviceCopyTime;
  exec = execTime;
  dToH = deviceToHostCopyTime;
}

#if STATS==1
void CudaKernelsBloom::processMessage(CudaOutbox *outbox) {
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
void CudaKernelsBloom::processMessage(CudaOutbox *outbox) {
  BFilter messageNames = copyMsgToDevice(outbox->message);
  computeResults(messageNames);
  getMatchingInterfaces(outbox->outgoingInterfaces);
}
#endif

BFilter CudaKernelsBloom::copyMsgToDevice(CudaMessage *message) {
  BFilter bf=0;
  int dest = 0;
  for (int i=0; i<message->numAttributes; i++) {
    add(bf, message->attributes[i].name);
    string_t name = message->attributes[i].name;
    if(nameDeviceBFConstr.find(name)==nameDeviceBFConstr.end()) {
      cerr << "Name: " << message->attributes[i].name << " not found during message processing" << endl;
      exit(1);
    }
    int numBFConstr = nameDeviceBFConstr[name].numBFConstr;
    int maxNumConstr = nameDeviceBFConstr[name].maxNumConstr;
    memcpy(hostInput[dest].bfConstr, nameDeviceBFConstr[name].bfConstr, sizeof(CudaBFConstr)*numBFConstr);
    hostInput[dest].numBFConstr = numBFConstr;
    hostInput[dest].maxNumConstr = maxNumConstr;
    hostInput[dest].value = message->attributes[i].value;
    dest++;
  }
  numValues = dest;
  if (dest>0) {
    int e = 0;
    e += cudaMemcpyToSymbolAsync(constInput, hostInput, (size_t) sizeof(CudaBFInputElem)*numValues);
    if (e>0) {
      cerr << " Memcpy error " << e << " during message processing " <<  endl;
      exit(1);
    }
  }
  return bf;
}

void CudaKernelsBloom::computeResults(BFilter messageNames) {
  int maxNumConstr = 0;
  for(int i=0; i<numValues; i++) {
    if(maxNumConstr<hostInput[i].maxNumConstr) maxNumConstr=hostInput[i].maxNumConstr;
  }
  evalConstraint<<<dim3(1+maxNumConstr/NUM_THREADS, NUM_DIFF_BF_IN_BFCONSTR, numValues), NUM_THREADS>>>(currentFiltersCount, filtersInfo, interfacesDevice, numFilters, numInterfaces, messageNames);
}

void CudaKernelsBloom::getMatchingInterfaces(set<int> &results) {
	int e = cudaMemcpyAsync(interfacesHost, interfacesDevice, (size_t) sizeof(unsigned char)*numInterfaces, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	//cudaMemsetAsync(currentFiltersCount, 0, (size_t) sizeof(unsigned int)*numFilters);
	//cudaMemsetAsync(interfacesDevice, 0, (size_t) sizeof(unsigned char)*numInterfaces);
	cleanCounters<<<numFilters/2048, NUM_THREADS>>>(currentFiltersCount, interfacesDevice, numFilters, numInterfaces);
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
