#ifndef EVALRUNNER_H_
#define EVALRUNNER_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <set>
#include <map>
#include "ParamHandler.h"
#include "WorkloadGenerator.h"
#include "../cuda/cudaTypes.h"
#include "../cuda/CudaKernels.h"
#include "../cuda/CudaKernelsNoDup.h"
#include "../cuda/CudaKernelsBloom.h"
#include "../cuda/CudaKernelsSimpleBloom.h"
#include "../common/Timer.h"
#include "../common/Translator.h"
#include "../common/Consts.h"

class EvalRunner {
public:
	EvalRunner();

	virtual ~EvalRunner();

	/**
	 * Run all test
	 */
	void runTests(int algo);

private:
	Timer *t;							// Timer
	Translator *translator;						// Translator
	ParamHandler *paramHandler;					// ParamHandler
	WorkloadGenerator *workloadGenerator;				// WorkloadGenerator

	/**
	 * Once parameters have been set, this function is called to actually execute tests
	 */
	void executeTest(int algo, int minSeed, int maxSeed, string &filename, double label);

	/** Scenarios **/
	void runDefaultScenario(int algo);
	void runZipfScenario(int algo);
	void runConstraintsPerFilterScenario(int algo);
	void runConstraintsPerFilterFixedScenario(int algo, int numConstraints);
	void runFiltersPerInterfaceScenario(int algo);
	void runInterfacesFixedScenario(int algo, int numConstraints);
	void runInterfacesScenario(int algo);
	void runNamesScenario(int algo);
	void runTypeScenario(int algo);
	void runAttributeScenario(int algo);
	void runConstraintsPerThread(int algo);
	void runOperatorScenario(int algo);
	void runValuesScenario(int algo);

	/** Write to file */
	void writeToFile(int seed, string &filename, double label, int numResults, double install, double value);
};

#endif /* EVALRUNNER_H_ */
