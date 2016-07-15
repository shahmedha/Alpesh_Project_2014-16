#ifndef SYSTEMEVALRUNNER_H_
#define SYSTEMEVALRUNNER_H_

#include <pthread.h>
#include <queue>
#include <map>
#include <set>
#include "../cuda/cudaTypes.h"
#include "../cuda/CudaKernels.h"
#include "../common/Timer.h"
#include "ParamHandler.h"
#include "FileReader.h"
#include "FileWriter.h"
#include "SocketReader.h"

/**
 * Data shared with the thread reading from the interfaces and writing on the input queue
 */
typedef struct InputDataStruct {
	std::queue<CudaOutbox *> *inputQueue;
	pthread_cond_t *inputCond;
	pthread_mutex_t *inputMutex;
	FileReader *fileReader;
	SocketReader *socketReader;
	int algo;
	int queueSize;
	bool finish;
} InputData;

/**
 * Data shared with the thread reading from the output queue and writing on interfaces
 */
typedef struct OutputDataStruct {
	std::queue<CudaOutbox *> *outputQueue;
	pthread_cond_t *outputCond;
	pthread_mutex_t *outputMutex;
	FileWriter *fileWriter;
	bool finish;
} OutputData;

class SystemEvalRunner {
public:
	/**
	 * Constructor
	 */
	SystemEvalRunner(int algo, int queueSize, int numMessages);

	/**
	 * Executes all defined tests
	 */
	void runTests();

	/**
	 * Destructor
	 */
	virtual ~SystemEvalRunner();

private:
	int algo;																		// Algorithm used during the test
	int queueSize;															// Size of the input queue
	int numMessages;														// Number of messages to generate
	Timer *t;																		// Timer
	ParamHandler *paramHandler;									// Parameter handler
	SocketReader *socketReader;									// Socket reader
	FileWriter *fileWriter;											// File writer
	FileReader *fileReader;											// File reader
	WorkloadGenerator *workloadGenerator;				// WorkloadGenerator
	Translator *translator;											// Translator

	std::queue<CudaOutbox *> *inputQueue;				// Input queue
	std::queue<CudaOutbox *> *outputQueue;			// Output queue
	pthread_cond_t *inputCond;									// Input condition variable
	pthread_cond_t *outputCond;									// Output condition variable
	pthread_mutex_t *inputMutex;								// Input mutex
	pthread_mutex_t *outputMutex;								// Output mutex

	/**
	 * Functions to set the parameters according to the test to be performed
	 */
	void runDefaultScenario();
	void runZipfScenario();
	void runNetworkScenario();

	/**
	 * Starts a single execution of a test
	 */
	void runTest(int minSeed, int maxSeed, string &filename, double label);

	/**
	 * Fills input files with serialized messages
	 */
	void fillInputFiles();

	/**
	 * Writes results to file
	 */
	void writeToFile(int seed, string &filename, double label, int numResults, int numMessages, double time);

};

#endif /* SYSTEMEVALRUNNER_H_ */
