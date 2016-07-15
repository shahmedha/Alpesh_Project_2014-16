#include "SystemEvalRunner.h"

using namespace std;

void * inputThreadFun(void *sharedData) {
	InputData *in = (InputData *) sharedData;
	int pos = 0;
	if (in->algo==6) {
		cout << " *** Waiting for client connection ... ";
		fflush(stdout);
		in->socketReader->openConnection();
		cout << "DONE! *** " << endl;
	}
	while (true) {
		CudaOutbox *outbox = NULL;
		if (in->algo==5) outbox = in->fileReader->readFromFile(0, pos);
		else if (in->algo==6) outbox = in->socketReader->readFromNetwork();
		if (outbox==NULL) {
			pthread_mutex_lock(in->inputMutex);
			in->finish = true;
			if (in->inputQueue->empty()) {
				pthread_cond_signal(in->inputCond);
			}
			pthread_mutex_unlock(in->inputMutex);
			pthread_exit(NULL);
		} else {
			pthread_mutex_lock(in->inputMutex);
			int currentSize = in->inputQueue->size();
			if (in->queueSize>0 && currentSize>in->queueSize) {
				delete outbox->message;
				delete outbox;
			} else {
				in->inputQueue->push(outbox);
				if (in->inputQueue->size()==1) {
					pthread_cond_signal(in->inputCond);
				}
			}
			pthread_mutex_unlock(in->inputMutex);
		}
	}
}

void * outputThreadFun(void *sharedData) {
	OutputData *out = (OutputData *) sharedData;
	while (true) {
		pthread_mutex_lock(out->outputMutex);
		if (out->outputQueue->empty() && ! out->finish) {
			pthread_cond_wait(out->outputCond, out->outputMutex);
		}
		if (out->finish && out->outputQueue->empty()) {
			pthread_mutex_unlock(out->outputMutex);
			pthread_exit(NULL);
		} else {
			CudaOutbox *outbox = out->outputQueue->front();
			out->outputQueue->pop();
			pthread_mutex_unlock(out->outputMutex);
			out->fileWriter->writeResultsToFile(outbox);
			delete outbox->message;
			delete outbox;
		}
	}
}

SystemEvalRunner::SystemEvalRunner(int algo, int queueSize, int numMessages) {
	this->algo = algo;
	this->queueSize = queueSize;
	this->numMessages = numMessages;
	t = new Timer();
	paramHandler = new ParamHandler();
	workloadGenerator = new WorkloadGenerator(paramHandler);
	translator = new Translator();
	fileWriter = new FileWriter(workloadGenerator, translator);
	fileReader = new FileReader();
	socketReader = new SocketReader();
	inputQueue = new queue<CudaOutbox *>;
	outputQueue = new queue<CudaOutbox *>;
	inputCond = new pthread_cond_t;
	pthread_cond_init(inputCond, NULL);
	outputCond = new pthread_cond_t;
	pthread_cond_init(outputCond, NULL);
	inputMutex = new pthread_mutex_t;
	pthread_mutex_init(inputMutex, NULL);
	outputMutex = new pthread_mutex_t;
	pthread_mutex_init(outputMutex, NULL);
}

SystemEvalRunner::~SystemEvalRunner() {
	delete t;
	delete paramHandler;
	delete workloadGenerator;
	delete translator;
	pthread_cond_destroy(inputCond);
	delete inputCond;
	pthread_cond_destroy(outputCond);
	delete outputCond;
	pthread_mutex_destroy(inputMutex);
	delete inputMutex;
	pthread_mutex_destroy(outputMutex);
	delete outputMutex;
	delete inputQueue;
	delete outputQueue;
	delete fileReader;
	delete fileWriter;
	if (algo==6) {
		socketReader->closeServerSocket();
	}
	delete socketReader;
}

void SystemEvalRunner::runTests() {
	if (algo==6) {
		cout << endl << " *** Binding server address *** " << endl;
		socketReader->bindAddress(9000);
		runNetworkScenario();
	} else {
		runDefaultScenario();
		//runZipfScenario();
	}
}

void SystemEvalRunner::fillInputFiles() {
	fileWriter->generateInterfaceFiles(1);
}

void SystemEvalRunner::runDefaultScenario() {
	paramHandler->resetToDefault();
	paramHandler->setNumMessages(numMessages);
	string filename = "Default";
	runTest(0, 9, filename, 3);
}

void SystemEvalRunner::runZipfScenario() {
	paramHandler->resetToDefault();
	paramHandler->setNumMessages(numMessages);
	paramHandler->setZipfNames(true);
	string filename = "Zipf";
	runTest(0, 9, filename, 3);
}

void SystemEvalRunner::runNetworkScenario() {
	paramHandler->resetToDefault();
	paramHandler->setNumMessages(numMessages);
	string filename = "Network";
	for (int i=1200; i>=20; ) {
		runTest(0, 9, filename, i);
		if (i>100) i-=100;
		else i-=20;
	}
}

void SystemEvalRunner::runTest(int minSeed, int maxSeed, string &filename, double label) {
	for (int s=minSeed; s<=maxSeed; s++) {
		srand(s);
		// TODO: check (ask Ale) if the two lines below are really required. They are not there in EvalRunner
		workloadGenerator->resetNames();
		workloadGenerator->resetStringValues();
		map<int, set<simple_filter *> > subscriptions;
		map<int, set<CudaFilter *> > cudaSubscriptions;

		if (algo==5) {
			cout << endl << " *** Generating Messages *** " << endl;
			fillInputFiles();
		}

		cout << endl << " *** Generating Subscriptions *** " << endl;
		workloadGenerator->generateSubscriptions(subscriptions);

		cout << endl << " *** Translating Subscriptions *** " << endl;
		translator->translatePredicate(subscriptions, cudaSubscriptions);

		cout << endl << " *** Deleting Subscriptions ***" << endl;
		for (map<int, set<simple_filter *> >::iterator it=subscriptions.begin(); it!=subscriptions.end(); ++it) {
			for (set<simple_filter *>::iterator it2=it->second.begin(); it2!=it->second.end(); ++it2) {
				simple_filter *filter = *it2;
				delete filter;
			}
		}

		CudaKernels *table = new CudaKernels();
		for (map<int, set<CudaFilter *> >::iterator it=cudaSubscriptions.begin(); it!=cudaSubscriptions.end(); ++it) {
			table->ifConfig(it->first, it->second);
		}
		cout << endl << " *** Consolidating Table *** " << endl;
		table->consolidate();

		// Threads
		pthread_t *inputThread = new pthread_t;
		pthread_t *outputThread = new pthread_t;

		// Input Data
		InputData *inputData = new InputData;
		inputData->fileReader = fileReader;
		inputData->socketReader = socketReader;
		inputData->finish = false;
		inputData->inputCond = inputCond;
		inputData->inputMutex = inputMutex;
		inputData->inputQueue = inputQueue;
		inputData->algo = algo;
		inputData->queueSize = queueSize;

		// Output Data
		OutputData *outputData = new OutputData;
		outputData->fileWriter = fileWriter;
		outputData->finish = false;
		outputData->outputCond = outputCond;
		outputData->outputMutex = outputMutex;
		outputData->outputQueue = outputQueue;

		int numMessages = 0;
		int numResults = 0;
		double time = 0;

		cout << endl << " *** Starting Test *** " << endl;
		pthread_create(outputThread, NULL, outputThreadFun, (void *) outputData);
		pthread_create(inputThread, NULL, inputThreadFun, (void *) inputData);

		t->start();
		while (true) {
			pthread_mutex_lock(inputData->inputMutex);
			if (inputData->inputQueue->empty() && ! inputData->finish) {
				pthread_cond_wait(inputData->inputCond, inputData->inputMutex);
			}
			if (inputData->inputQueue->empty() && inputData->finish) {
				pthread_mutex_unlock(inputData->inputMutex);
				pthread_mutex_lock(outputData->outputMutex);
				outputData->finish = true;
				pthread_cond_signal(outputData->outputCond);
				pthread_mutex_unlock(outputData->outputMutex);
				break;
			} else {
				CudaOutbox *outbox = inputData->inputQueue->front();
				inputData->inputQueue->pop();
				pthread_mutex_unlock(inputData->inputMutex);
				table->processMessage(outbox);
				numResults += outbox->outgoingInterfaces.size();
				numMessages++;
				pthread_mutex_lock(outputData->outputMutex);
				outputData->outputQueue->push(outbox);
				pthread_cond_signal(outputData->outputCond);
				pthread_mutex_unlock(outputData->outputMutex);
			}
		}

		pthread_join(*inputThread, NULL);
		pthread_join(*outputThread, NULL);

		time += t->stop();
		double avgTime = time/numMessages;
		writeToFile(s, filename, label, numResults, numMessages, avgTime);
		cout << " *** Test Executed *** " << endl;
		cout << " *** Processed Messages: " << numMessages << " *** " << endl;
		cout << " *** Generated Results: " << numResults << " *** " << endl;
		cout << " *** Processing Time: " << avgTime << " *** " << endl;
#if STATS==1
		double input, exec, res;
		table->getStats(input, exec, res);
		cout << " *** Preparing input / Processing / Copy Results: " << input << " / " << exec << " / " << res << " *** " << endl;
#endif
		cout << endl << " *** Deleting table *** " << endl;
		delete table;
		delete inputData;
		delete outputData;
		delete inputThread;
		delete outputThread;
	}
}


void SystemEvalRunner::writeToFile(int seed, string &filename, double label, int numResults, int numMessages, double time) {
	string algo;
	stringstream stream;
	stream << seed;
	string stringSeed = stream.str();
	string name = "Results/" + filename + "_system_cuda_" + stringSeed;
	ofstream file;
	file.open(name.data(), ios::app);
	file << label << "\t" << time << "\t" << numResults << "\t" << numMessages << endl;
	file.close();
}
