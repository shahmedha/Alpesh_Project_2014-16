#ifndef CUDATYPES_H_
#define CUDATYPES_H_

#include <set>
#include "../common/Consts.h"
#include "../sff/siena/types.h"

enum TypeEnum {
	INT=0,
	STRING=1
};

typedef unsigned char Type;

enum OpEnum {
	EQ=0,		// Equals (numbers+strings)
	LT=1,		// Less then (numbers)
	GT=2,		// Greater than (numbers)
	DF=3,		// Different From (numbers+strings)
	IN=4,		// Contains (strings)
	PF=5		// Prefix (strings)
};

typedef unsigned char Op;

typedef struct CudaValueStruct {
	int intVal;
	char stringVal[STRING_VAL_LEN];
	Type type;
} CudaValue;

typedef struct CudaAttributeStruct {
	char name[NAME_LEN];
	CudaValue value;
} CudaAttribute;

typedef struct CudaMessageStruct {
	CudaAttribute attributes[MAX_ATTR_NUM];
	int numAttributes;
} CudaMessage;

typedef struct CudaConstraintStruct {
	char name[NAME_LEN];
	Op op;
	CudaValue value;
} CudaConstraint;

typedef struct CudaFilterStruct {
	CudaConstraint constraints[MAX_CONSTR_NUM];
	int numConstraints;
	int interface;
} CudaFilter;

typedef struct CudaOutboxStruct {
	CudaMessage *message;
	std::set<int> outgoingInterfaces;
} CudaOutbox;

/* -------------- ALGORITHM SPECIFIC DATA STRUCTURES --------------- */

typedef struct FilterInfoStruct {
	unsigned char numConstraints;
	unsigned char interface;
} FilterInfo;

typedef struct IntCudaConstraintStruct {
        int value;
} IntCudaConstraint;

typedef struct StringCudaConstraintStruct {
        char value[STRING_VAL_LEN];
} StringCudaConstraint;

typedef struct CudaInputElemStruct {
        CudaValue value;
        void *constrVal;     // pointer to the right row of constraint values
        Op *constrOp;        // pointer to the right row of constraint operands
        int *filterIdx;      // pointer to the right row of filter indexes (for each constraint this is the index of the filter it belongs to) 
        int numConstraints;
} CudaInputElem;

/* ----------- bloom ----------- */

#define NUM_DIFF_BF_IN_BFCONSTR 32
#define MAX_NUM_NAMES_IN_BF 1

typedef unsigned int BFilter;

typedef struct CudaBFConstrStruct {
        BFilter filterNames;
        Op *constrOp;        // pointer to the right row of constraint operands
        void *constrVal;     // pointer to the right row of constraint values
        int *filterIdx;      // pointer to the right row of filter indexes (for each constraint this is the index of the filter it belongs to) 
        int numConstr;
} CudaBFConstr;

typedef struct CudaBFInputElemStruct {
        CudaValue value;
        CudaBFConstr bfConstr[NUM_DIFF_BF_IN_BFCONSTR];
        int numBFConstr;
        int maxNumConstr;
} CudaBFInputElem;

typedef struct {
  CudaBFConstr *bfConstr;  // max NUM_DIFF_BF_IN_BFCONSTR elements
  int numBFConstr;
  int maxNumConstr;
} BFConstrList;

/* ----------- simplified bloom ----------- */

typedef struct CudaSBFInputElemStruct {
        CudaValue value;
        void *constrVal;      // pointer to the right row of constraint values
        Op *constrOp;         // pointer to the right row of constraint operands
        int *filterIdx;       // pointer to the right row of filter indexes (for each constraint this is the index of the filter it belongs to) 
        BFilter *filterNames; // pointer to the right row of filter names (for each constraint this is the BFilter encoding the names of the filter it belongs to) 
        int numConstraints;
} CudaSBFInputElem;

#endif /* CUDATYPES_H_ */
