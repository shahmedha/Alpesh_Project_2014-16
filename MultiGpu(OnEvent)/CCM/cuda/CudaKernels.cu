#include "CudaKernels.h"

#define NUM_THREADS 256


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

static __global__ void evalConstraint(unsigned char *filtersCount, const FilterInfo *filterInfo, unsigned char *interfaces, const int numFilters, const int numInterfaces, int attributeIdx, CudaInputElem *constInput) {
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
  int filterIndex = inputElem.filterIdx[constraintsIndex];
  filtersCount[filterIndex]++;
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

CudaKernels::CudaKernels() {
  numInterfaces = 0;
  numFilters = 0;
  consolidated = false;
  hostToDeviceCopyTime = 0;
  execTime = 0;
  deviceToHostCopyTime = 0;
}

CudaKernels::~CudaKernels() {
       
  if (consolidated) {
  
  for(int d=0;d<ngpus;d++){
   cudaSetDevice(d);
    for (map<string_t, void *>::iterator it=nameDeviceConstrVal[d].begin(); it!=nameDeviceConstrVal[d].end(); ++it) {
    
      void *constrPtr = it->second;
     
      cudaFree(constrPtr);
       //cout<<constrPtr<<"\nCudaKernel Destructor\n"<<endl;
     // cout<<constrPtr<<"\nCudaKernel Destructor\n"<<endl;
    }
    
    for (map<string_t, Op *>::iterator it=nameDeviceConstrOp[d].begin(); it!=nameDeviceConstrOp[d].end(); ++it) {
      Op *constrPtr = it->second;
      cudaFree(constrPtr);
    }
    for (map<string_t, int *>::iterator it=nameDeviceFilterIdx[d].begin(); it!=nameDeviceFilterIdx[d].end(); ++it) {
      int *filterIdxPtr = it->second;
      cudaFree(filterIdxPtr);
    }
    cudaFreeHost(hostInput[d]);
    cudaFree(currentFiltersCount[d]);
    cudaFree(filtersInfo[d]);
    cudaFree(interfacesDevice[d]);
    
    delete interfacesHost[d];
  }
  for (map<int, set<CudaFilter *> >::iterator it=hostFilters.begin(); it!=hostFilters.end(); ++it) {
    for (set<CudaFilter *>::iterator it2=it->second.begin(); it2!=it->second.end(); ++it2) {
      CudaFilter *filter = *it2;
      delete filter;
    }
  }
  }
}

void CudaKernels::ifConfig(int interfaceId, set<CudaFilter *> &filters) {
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

void all_host_allocation_filters()
{



}
void CudaKernels::consolidate() {

    /// host structures
     map<string_t, int> currentNumConstraints;
        map<string_t, void *> nameHostConstrVal;
        map<string_t, Op *> nameHostConstrOp;
     map<string_t, int *> nameHostFilterIdx;
    for (map<string_t, int>::iterator it=numConstraints.begin(); it!=numConstraints.end(); ++it) {
            string_t name = it->first;
            int num = it->second;
            void *hostConstrValPtr;
            if(nameType[name]==INT) {
              hostConstrValPtr = malloc(sizeof(IntCudaConstraint)*num);
            } else {
              hostConstrValPtr = malloc(sizeof(StringCudaConstraint)*num);
            }
            nameHostConstrVal.insert(make_pair(name, hostConstrValPtr));
            Op* hostConstrOpPtr;
            hostConstrOpPtr = (Op *)malloc(sizeof(Op)*num);
            nameHostConstrOp.insert(make_pair(name, hostConstrOpPtr));
            currentNumConstraints.insert(make_pair(name, 0));
            int *hostFilterIdxPtr;
             hostFilterIdxPtr = (int *)malloc(sizeof(int)*num);
             nameHostFilterIdx.insert(make_pair(name, hostFilterIdxPtr));
    }
    /// initialize the nameHostConstrVal, nameHostConstrOp, nameHostFilterIdx, and hostFiltersInfo structures
          ///(to be copied into the corresponding structures in device later)
          int filterId = 0;
          FilterInfo *hostFiltersInfo = (FilterInfo *) malloc(sizeof(FilterInfo)*numFilters);
          for (map<int, set<CudaFilter *> >::iterator it=hostFilters.begin(); it!=hostFilters.end(); ++it) {
            int interfaceId = it->first;
            for (set<CudaFilter *>::iterator it2=it->second.begin(); it2!=it->second.end(); ++it2) {
              CudaFilter *filter = *it2;
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
              }
              hostFiltersInfo[filterId].numConstraints = filter->numConstraints;
              hostFiltersInfo[filterId].interface = interfaceId;
              filterId++;
            }
          }

    /// device functions copy

    cudaGetDeviceCount(&ngpus);
    nameDeviceConstrVal.resize(ngpus);
    nameDeviceConstrOp.resize(ngpus);
    nameDeviceFilterIdx.resize(ngpus) ;
   // cudaStreams = = (cudaStream_t *)malloc(sizeof(cudaStream_t) * ngpus);

    hostInput = (CudaInputElem **)malloc(sizeof(CudaInputElem *)*ngpus);
    interfacesHost = (unsigned char **)malloc(sizeof(unsigned char *)*ngpus);
    interfacesDevice = (unsigned char **)malloc(sizeof(unsigned char *)*ngpus);
    currentFiltersCount = (unsigned char **)malloc(sizeof(unsigned char *)*ngpus);
    filtersInfo = (FilterInfo **)malloc(sizeof(FilterInfo *)*ngpus);
    constInput = (CudaInputElem **)malloc(sizeof(CudaInputElem * )*ngpus);
    cout<<"No. of Cuda Devices "<<ngpus<<endl;

    /// multiple devices
     int e = 0;
     
          int allocSize = 0;
    for(int device = 0 ; device < ngpus ; device++){
                
          cudaSetDevice(device);
         
           //static __constant__ constInput[i]=[MAX_ATTR_NUM];
            e+=cudaMalloc((void**)&constInput[device] , (size_t)sizeof(CudaInputElem)*MAX_ATTR_NUM);
       //   cudaStreamCreate(&cudaStreams[i]);///not needed
          /// host input data structures... to be copied to Gpu
            e += cudaMallocHost((void**) &hostInput[device], (size_t) sizeof(CudaInputElem)*MAX_ATTR_NUM);

    ///interface array on host like pinned memory
         interfacesHost[device] = (unsigned char *) malloc( (size_t) sizeof(unsigned char)*numInterfaces);
            // allocate memory on device and host

          numInterfaces = hostFilters.size();
          allocSize += sizeof(CudaInputElem)*MAX_ATTR_NUM;  // allocated into constant memory (see static variable at the beginning of file)

          e += cudaMalloc((void**) &interfacesDevice[device], (size_t) sizeof(unsigned char)*numInterfaces);
          allocSize += sizeof(unsigned char)*numInterfaces;
                
               
          /// allocation for host and device data structuers . host datastructure pinned memory to copy
          /// map stores pointers to addresses
          for (map<string_t, int>::iterator it=numConstraints.begin(); it!=numConstraints.end(); ++it) {
            string_t name = it->first;
            int num = it->second;
            void *constrValPtr;
            if(nameType[name]==INT) {
              e += cudaMalloc((void**) &constrValPtr, (size_t) sizeof(IntCudaConstraint)*num);
              allocSize += sizeof(IntCudaConstraint)*num;
            } else {
              e += cudaMalloc((void**) &constrValPtr, (size_t) sizeof(StringCudaConstraint)*num);
              allocSize += sizeof(StringCudaConstraint)*num;
            }
            nameDeviceConstrVal[device].insert(make_pair(name, constrValPtr));
            Op *constrOpPtr;
            e+= cudaMalloc((void**) &constrOpPtr, (size_t) sizeof(Op)*num);https://accounts.google.com/ServiceLogin?service=mail&passive=true&rm=false&continue=https://mail.google.com/mail/&ss=1&scc=1&ltmpl=default&ltmplcache=2&emr=1&osid=1
            allocSize += sizeof(Op)*num;
            nameDeviceConstrOp[device].insert(make_pair(name, constrOpPtr));
            int *filterIdxPtr;
            e+= cudaMalloc((void**) &filterIdxPtr, (size_t) sizeof(int)*num);
            allocSize += sizeof(int)*num;
            nameDeviceFilterIdx[device].insert(make_pair(name, filterIdxPtr));
          }
          e += cudaMalloc((void**) &currentFiltersCount[device], (size_t) sizeof(unsigned char)*numFilters);
          allocSize += sizeof(unsigned char)*numFilters;
          e += cudaMalloc((void**) &filtersInfo[device], (size_t) sizeof(FilterInfo)*numFilters);
          allocSize += sizeof(FilterInfo)*numFilters;
          if (e>0) {
            cerr << " Allocation error " << e << endl;
            exit(1);
          }

    }
    
    for(int device=0; device < ngpus ; device ++){
            cudaSetDevice(device);
       //   cudaStreamCreate(&cudaStreams[i]);///not needed
          /// initialize the device memory
          void *host;
          for (map<string_t, void *>::iterator it=nameHostConstrVal.begin(); it!=nameHostConstrVal.end(); ++it) {
            string_t name = it->first;
            host = it->second;
            void *device_add = nameDeviceConstrVal[device][name];
            int size = numConstraints[name];
            if(nameType[name]==INT) {
              e += cudaMemcpyAsync(device_add, host, sizeof(IntCudaConstraint)*size, cudaMemcpyHostToDevice);
            } else {
              e += cudaMemcpyAsync(device_add, host, sizeof(StringCudaConstraint)*size, cudaMemcpyHostToDevice);
            }
            //cudaDeviceSynchronize();
            //
          }
          //free(host);
          Op *host1;
          for (map<string_t, Op *>::iterator it=nameHostConstrOp.begin(); it!=nameHostConstrOp.end(); ++it) {
            string_t name = it->first;
            host1 = it->second;
            Op *device_add = nameDeviceConstrOp[device][name];
            int size = numConstraints[name];
            e += cudaMemcpyAsync(device_add, host1, sizeof(Op)*size, cudaMemcpyHostToDevice);
            //cudaDeviceSynchronize();

          }
          //free(host1);
          int *host2;
          for (map<string_t, int *>::iterator it=nameHostFilterIdx.begin(); it!=nameHostFilterIdx.end(); ++it) {
            string_t name = it->first;
            host2 = it->second;
            int *device_add = nameDeviceFilterIdx[device][name];
            int size = numConstraints[name];
            e += cudaMemcpyAsync(device_add, host2, sizeof(int)*size, cudaMemcpyHostToDevice);
            //cudaDeviceSynchronize();

          }
         // free(host2);
          e += cudaMemcpyAsync(filtersInfo[device], hostFiltersInfo, (size_t) sizeof(FilterInfo)*numFilters, cudaMemcpyHostToDevice);
          cudaMemsetAsync(currentFiltersCount[device], 0, (size_t) sizeof(unsigned char)*numFilters);
          cudaMemsetAsync(interfacesDevice[device], 0, (size_t) sizeof(unsigned char)*numInterfaces);
          cudaDeviceSynchronize();
          consolidated = true;
          if (e>0) {
            cerr << " Memcpy error " << e << " during consolidation " <<  endl;
            exit(1);
          }


          // set up the runtime to optimize performance
          //cudaFuncSetCacheConfig(evalConstraint, cudaFuncCachePreferL1);
          cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    }
   
    for(int device=0;device<ngpus;device++){
        cudaSetDevice(device);
        cudaDeviceSynchronize();
    }

          int totConstr=0;
          for(map<string_t,int>::iterator it=numConstraints.begin(); it!=numConstraints.end(); ++it) {
            totConstr+=it->second;
          }
          cout << endl << " ### " << totConstr << " constraints allocated ### " << endl;
          cout << endl << " ### " << allocSize << " bytes allocated on device ### " << endl;
          cout << endl << "#####################" << endl;
    free(hostFiltersInfo);

}
int CudaKernels::getGpuCount(){
    return ngpus;
}
void CudaKernels::getStats(double &hToD, double &exec, double &dToH) {
  hToD = hostToDeviceCopyTime;
  exec = execTime;
  dToH = deviceToHostCopyTime;
}

#if STATS==1
void CudaKernels::processMessage(CudaOutbox *outbox,int dev_no) {
  Timer t;
  t.start();
  dev_no%=ngpus;

  cudaSetDevice(dev_no);  /// cuda set device

  int maxConstr = copyMsgToDevice(outbox->message,dev_no);
  //cudaDeviceSynchronize(); // TODO: remove
  hostToDeviceCopyTime += t.stop();
  if (maxConstr>0) {
    t.start();
    for(int i=0; i<numValues; i++) {
	  evalConstraint<<<hostInput[dev_no][i].numConstraints/NUM_THREADS+1, NUM_THREADS>>>(currentFiltersCount[dev_no], filtersInfo[dev_no], interfacesDevice[dev_no], numFilters, numInterfaces, i,constInput[dev_no]);
	}
	summarize<<<numFilters/2048, NUM_THREADS>>>(currentFiltersCount[dev_no], filtersInfo[dev_no], interfacesDevice[dev_no], numFilters, numInterfaces);
//    computeResults(maxConstr,dev_no);
    //cudaDeviceSynchronize(); // TODO: remove
    execTime += t.stop();
    //t.start();
    //getMatchingInterfaces(outbox->outgoingInterfaces,dev_no);
    //cudaDeviceSynchronize(); // TODO: remove
    //deviceToHostCopyTime += t.stop();
  }
}
#elif STATS==0
void CudaKernels::processMessage(CudaOutbox *outbox) {
  int maxConstr = copyMsgToDevice(outbox->message);
  if (maxConstr>0) {
    computeResults(maxConstr);
    getMatchingInterfaces(outbox->outgoingInterfaces);
  }
}
#endif

int CudaKernels::copyMsgToDevice(CudaMessage *message,int dev_no) {
  int dest = 0;
  int maxConstr = 0;
  for (int i=0; i<message->numAttributes; i++) {
    string_t name = message->attributes[i].name;
    map<string_t, void *>::iterator it = nameDeviceConstrVal[dev_no].find(name);
    if(it==nameDeviceConstrVal[dev_no].end()) {
      cerr << "Name: ";
      for(int i=0; i<name.length(); i++) cerr << name[i];
      cerr << " not found during message processing" << endl;
      exit(1);
    }
    hostInput[dev_no][dest].constrVal = it->second;
    map<string_t, Op *>::iterator it1 = nameDeviceConstrOp[dev_no].find(name);
    if(it1==nameDeviceConstrOp[dev_no].end()) {
      cerr << "Name: ";
      for(int i=0; i<name.length(); i++) cerr << name[i];
      cerr << " not found during message processing" << endl;
      exit(1);
    }
    hostInput[dev_no][dest].constrOp = it1->second;
    map<string_t, int *>::iterator it2 = nameDeviceFilterIdx[dev_no].find(name);
    if(it2==nameDeviceFilterIdx[dev_no].end()) {
      cerr << "Name: ";
      for(int i=0; i<name.length(); i++) cerr << name[i];
      cerr << " not found during message processing" << endl;
      exit(1);
    }
    hostInput[dev_no][dest].filterIdx = it2->second;
    hostInput[dev_no][dest].numConstraints = numConstraints[name];
    if (hostInput[dev_no][dest].numConstraints>maxConstr) maxConstr = hostInput[dev_no][dest].numConstraints;
    hostInput[dev_no][dest].value = message->attributes[i].value;
    dest++;
  }
  numValues = dest;
  if (dest>0) {
    int e = 0;
    e += cudaMemcpyAsync(constInput[dev_no], hostInput[dev_no], (size_t) sizeof(CudaInputElem)*numValues,cudaMemcpyHostToDevice);
    if (e>0) {
      cerr << " Memcpy error " << e << " during message processing " <<  endl;
      exit(1);
    }
  }
  return maxConstr;
}

void CudaKernels::computeResults(int maxConstr,int dev_no) {
	//int numBlocksX = 1+maxConstr/NUM_THREADS;
	//dim3 numBlocks = dim3(numBlocksX);
	for(int i=0; i<numValues; i++) {
	  evalConstraint<<<hostInput[dev_no][i].numConstraints/NUM_THREADS+1, NUM_THREADS>>>(currentFiltersCount[dev_no], filtersInfo[dev_no], interfacesDevice[dev_no], numFilters, numInterfaces, i,constInput[dev_no]);
	}
	summarize<<<numFilters/2048, NUM_THREADS>>>(currentFiltersCount[dev_no], filtersInfo[dev_no], interfacesDevice[dev_no], numFilters, numInterfaces);
}

void CudaKernels::getMatchingInterfaces(set<int> &results,int dev_no) {
    cudaSetDevice(dev_no);
    Timer t;
    t.start();
	int e = cudaMemcpyAsync(interfacesHost[dev_no], interfacesDevice[dev_no], (size_t) sizeof(unsigned char)*numInterfaces, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	deviceToHostCopyTime += t.stop();
	cudaMemsetAsync(currentFiltersCount[dev_no], 0, (size_t) sizeof(unsigned char)*numFilters);
	cudaMemsetAsync(interfacesDevice[dev_no], 0, (size_t) sizeof(unsigned char)*numInterfaces);
	//cudaDeviceReset();
	//cleanCounters<<<numFilters/2048, NUM_THREADS>>>(currentFiltersCount, interfacesDevice, numFilters, numInterfaces);
	if (e>0) {
		cerr << " Memcpy error " << e << " while copying matching interfaces " <<  endl;
		exit(1);
	}
	results.clear();
	for (int i=0; i<numInterfaces; i++) {
		if (interfacesHost[dev_no][i]!=0) {
			results.insert(i);
		}
	}
}
