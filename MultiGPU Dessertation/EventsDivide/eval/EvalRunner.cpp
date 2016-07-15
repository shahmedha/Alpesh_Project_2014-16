#include "EvalRunner.h"

EvalRunner::EvalRunner() {
	t = new Timer();
	translator = new Translator();
	paramHandler = new ParamHandler();
	workloadGenerator = new WorkloadGenerator(paramHandler);
}

EvalRunner::~EvalRunner() {

	delete t;
	delete translator;
	delete paramHandler;
	delete workloadGenerator;
}

void EvalRunner::runTests(int algo) {
  runDefaultScenario(algo);
    //runAttributeScenario(algo);
    //runInterfacesScenario(algo);
     //runFiltersPerInterfaceScenario(algo);
     //runConstraintsPerFilterScenario(algo);
/*  runZipfScenario(algo);
  runConstraintsPerFilterScenario(algo);
  runConstraintsPerFilterFixedScenario(algo, 200000);
  runConstraintsPerFilterFixedScenario(algo, 1000000);
  runConstraintsPerFilterFixedScenario(algo, 5000000);

  //runInterfacesFixedScenario(algo, 200000);
  //runInterfacesFixedScenario(algo, 1000000);
  //runInterfacesFixedScenario(algo, 5000000);

  runNamesScenario(algo);
  //runTypeScenario(algo);
  //runConstraintsPerThread(algo);
  runOperatorScenario(algo);
  runValuesScenario(algo);*/
}

void EvalRunner::runDefaultScenario(int algo) {
	string name = "Default";
	paramHandler->resetToDefault();
	executeTest(algo, 0, 9, name, 2);
}

void EvalRunner::runZipfScenario(int algo) {
	string name = "Zipf";
	paramHandler->resetToDefault();
	paramHandler->setZipfNames(true);
	executeTest(algo, 0, 9, name, 2);
}

void EvalRunner::runValuesScenario(int algo) {
  string name = "Values";
  paramHandler->resetToDefault();
  for (int i=1; i<=200; ) {
    paramHandler->setNumAttVal(i*100);
    executeTest(algo, 0, 9, name, i);
    if (i<10) i+=3;
    else i+=10;
  }
}

void EvalRunner::runConstraintsPerFilterScenario(int algo) {
	string name = "ConstraintsPerFilter";
	paramHandler->resetToDefault();
	for (int constr=1; constr<=9; constr++) {
		paramHandler->setMinConstr(constr-1);
		paramHandler->setMaxConstr(constr+1);
		if (constr==1) {
			paramHandler->setMinConstr(1);
			paramHandler->setMaxConstr(1);
		}
		executeTest(algo, 0, 9, name, constr);
	}
}

void EvalRunner::runConstraintsPerFilterFixedScenario(int algo, int numConstraints) {
	stringstream stream;
	stream << numConstraints;
	string stringConstraints = stream.str();
	string name = "ConstraintsPerFilterFixed_" + stringConstraints;
	paramHandler->resetToDefault();
	for (int constr=1; constr<=9; constr++) {
		int numFilters = numConstraints/(constr*paramHandler->getNumIf());
		paramHandler->setMinFilters(numFilters);
		paramHandler->setMaxFilters(numFilters);
		paramHandler->setMinConstr(constr-1);
		paramHandler->setMaxConstr(constr+1);
		if (constr==1) {
			paramHandler->setMinConstr(1);
			paramHandler->setMaxConstr(1);
		}
		executeTest(algo, 0, 9, name, constr);
	}
}

void EvalRunner::runFiltersPerInterfaceScenario(int algo) {
	string name = "FiltersPerInterface";
	paramHandler->resetToDefault();
	for (int filter=25000; filter<=250000; ) {
		int filterPerc = filter/10;
		paramHandler->setMinFilters(filter-filterPerc);
		paramHandler->setMaxFilters(filter+filterPerc);
		executeTest(algo, 0, 0, name, filter);
		if (filter<1000) filter+=300;
		else if (filter<10000) filter+=3000;
		else if (filter<100000) filter+=50000;
		else filter+=50000;
	}
}

void EvalRunner::runInterfacesFixedScenario(int algo, int numConstraints) {
	stringstream stream;
	stream << numConstraints;
	string stringConstraints = stream.str();
	string name = "InterfacesFixed_" + stringConstraints;
	paramHandler->resetToDefault();
	for (int i=10; i<=100; i+=10) {
		paramHandler->setNumIf(i);
		int numFilters = numConstraints/i;
		paramHandler->setMinFilters(numFilters);
		paramHandler->setMaxFilters(numFilters);
		executeTest(algo, 0, 9, name, i);
	}
}

void EvalRunner::runInterfacesScenario(int algo) {
	string name = "Interfaces";
	paramHandler->resetToDefault();
	for (int i=10; i<=100; i+=10) {
		paramHandler->setNumIf(i);
		executeTest(algo, 0, 0, name, i);
	}
}

void EvalRunner::runNamesScenario(int algo) {
	string name = "Names";
	paramHandler->resetToDefault();
	for (int i=10; i<=1000; ) {
		paramHandler->setNumNames(i);
		paramHandler->setNumIntNames(i);
		workloadGenerator->resetNames();
		executeTest(algo, 0, 9, name, i);
		if (i<100) i+=10;
		else i+=100;
	}
}

void EvalRunner::runTypeScenario(int algo) {
	string name = "Type";
	paramHandler->resetToDefault();
	workloadGenerator->resetNames();
	for (int i=0; i<=100; i+=10) {
		paramHandler->setNumIntNames(i);
		workloadGenerator->resetStringValues();
		executeTest(algo, 0, 9, name, i);
	}
}

void EvalRunner::runAttributeScenario(int algo) {
	string name = "Attribute";
	paramHandler->resetToDefault();
	workloadGenerator->resetNames();
	workloadGenerator->resetStringValues();
	for (int i=1; i<=9; i++) {
		paramHandler->setMinAttr(i);
		paramHandler->setMaxAttr(i);
		executeTest(algo, 0, 0, name, i);
	}
}

void EvalRunner::runConstraintsPerThread(int algo) {
	string name = "ConstraintsPerThread";
	paramHandler->resetToDefault();
	for (int i=1; i<=20; i++) {
		paramHandler->setConstraintsPerThread(i);
		executeTest(algo, 0, 9, name, i);
	}
}

void EvalRunner::runOperatorScenario(int algo) {
  string name = "Operator";
  paramHandler->resetToDefault();
  for (int i=0; i<=100; i+=10) {
    paramHandler->setPercIntEq(i);
    int perc = (100-i)/3;
    paramHandler->setPercIntGt(perc);
    paramHandler->setPercIntLt(perc);
    paramHandler->setPercIntDf(perc);
    executeTest(algo, 0, 9, name, i);
  }
}

void EvalRunner::executeTest(int algo, int minSeed, int maxSeed, string &filename, double label) {
        //minSeed to Maxseed
	for (int s=minSeed; s<=minSeed; s++) {
		cout << endl << " #### Running test " << filename << " " << label << " seed=" << s << " #### " << endl;
		srand(s);
		set<simple_message *> messages;
		map<int, set<simple_filter *> > subscriptions;
		set<CudaMessage *> cudaMessages;
		map<int, set<CudaFilter *> > cudaSubscriptions;

		cout << endl << " *** Generating Messages *** " << endl;
		workloadGenerator->generateMessages(messages);

		cout << endl << " *** Generating Subscriptions *** " << endl;
		workloadGenerator->generateSubscriptions(subscriptions);

		cout << endl << " *** Translating Messages *** " << endl;
		translator->translateMessages(messages, cudaMessages);

		cout << endl << " *** Translating Subscriptions *** " << endl;
		translator->translatePredicate(subscriptions, cudaSubscriptions);

		if (algo==1) {
			cout << endl << " *** Installing Subscriptions *** " << endl;
			CudaKernels *cudaKernels = new CudaKernels();
			t->start();
			for (map<int, set<CudaFilter *> >::iterator it=cudaSubscriptions.begin(); it!=cudaSubscriptions.end(); ++it) {
				cudaKernels->ifConfig(it->first, it->second);
			}
			cudaKernels->consolidate();
			double installTime = t->stop();
                  int ngpus = cudaKernels->getGpuCount();
			cout << endl << " *** Deleting subscriptions *** " << endl;
			for (map<int, set<simple_filter *> >::iterator it=subscriptions.begin(); it!=subscriptions.end(); ++it) {
				for (set<simple_filter *>::iterator it2=it->second.begin(); it2!=it->second.end(); ++it2) {
					simple_filter *filter = *it2;
					delete filter;
				}
			}

			int numMessages = 0;
			int numResults = 0;
			double time = 0;
			cout << endl << " *** Running Test *** " << endl;
                           CudaOutbox **outbox = new CudaOutbox*[ngpus];
                            for(int j=0;j<ngpus;j++)
                                       outbox[j]= new CudaOutbox;

                                       int cn=0;
                                       int temp_print = 0;
                                       freopen("oo.txt","w",stderr);
                                       for (set<CudaMessage *>::iterator tmp,it=cudaMessages.begin(); it!=cudaMessages.end(); ++it) {

                                        int dev=0;


                                        tmp=it;

                                        t->start();
                                        while(tmp!=cudaMessages.end()&&dev<ngpus){
                                            outbox[dev]->message = *tmp;

                                            cudaKernels->processMessage(outbox[dev],dev);
                                            numMessages++;
                                            tmp++;
                                            dev++;
                                        }
                                        tmp=it;

                                        dev=0;
                                        while(tmp!=cudaMessages.end()&&dev<ngpus){
                                              //  cout<<"g\n";
                                            cudaKernels->getMatchingInterfaces(outbox[dev]->outgoingInterfaces,dev%ngpus);
                                            cerr<<"event no "<<temp_print++<<" dev_no "<<dev<<" size "<<outbox[dev]->outgoingInterfaces.size()<<endl;
                                            numResults += outbox[dev]->outgoingInterfaces.size();
                                            tmp++;
                                            dev++;
                                        }
                                        tmp--;
                                        it=tmp;

				                        time += t->stop();
				                        cn++;

				         //
                        }
			 cout<<"\nTotal Time Requried to process "<<numMessages<<" : "<<time<<endl;
			double avgTime = time/numMessages;
			//writeToFile(s, filename, label, numResults, installTime, avgTime);
			cout << " *** Test Executed *** " << endl;
			cout << " *** Generated Results: " << numResults << " *** " << endl;
			cout << " *** Processing Time: " << avgTime << " *** " << endl;
#if STATS==1
			double hToD, exec, dToH;
			cudaKernels->getStats(hToD, exec, dToH);
			cout << " *** Copy Data / Processing / Copy Results: " << hToD << " / " << exec << " / " << dToH << " *** " << endl;
			cout << endl << " *** Deleting table *** " << endl;
#endif
			delete cudaKernels;

		} /*else if (algo==2) {
			cout << endl << " *** Installing Subscriptions *** " << endl;
			CudaKernelsNoDup *cudaKernels = new CudaKernelsNoDup();
			t->start();
			for (map<int, set<CudaFilter *> >::iterator it=cudaSubscriptions.begin(); it!=cudaSubscriptions.end(); ++it) {
				cudaKernels->ifConfig(it->first, it->second);
			}
			cudaKernels->consolidate();
			double installTime = t->stop();

			cout << endl << " *** Deleting subscriptions *** " << endl;
			for (map<int, set<simple_filter *> >::iterator it=subscriptions.begin(); it!=subscriptions.end(); ++it) {
				for (set<simple_filter *>::iterator it2=it->second.begin(); it2!=it->second.end(); ++it2) {
					simple_filter *filter = *it2;
					delete filter;
				}
			}

			int numMessages = 0;
			int numResults = 0;
			double time = 0;
			cout << endl << " *** Running Test *** " << endl;
			for (set<CudaMessage *>::iterator it=cudaMessages.begin(); it!=cudaMessages.end(); ++it) {
				CudaOutbox *outbox = new CudaOutbox;
				outbox->message = *it;
				t->start();
				cudaKernels->processMessage(outbox);
				time += t->stop();
				numResults += outbox->outgoingInterfaces.size();
				numMessages++;
				delete outbox;
			}
			double avgTime = time/numMessages;
			writeToFile(s, filename, label, numResults, installTime, avgTime);
			cout << " *** Test Executed *** " << endl;
			cout << " *** Generated Results: " << numResults << " *** " << endl;
			cout << " *** Processing Time: " << avgTime << " *** " << endl;
#if STATS==1
			double hToD, exec, dToH;
			cudaKernels->getStats(hToD, exec, dToH);
			cout << " *** Copy Data / Processing / Copy Results: " << hToD << " / " << exec << " / " << dToH << " *** " << endl;
			cout << endl << " *** Deleting table *** " << endl;
#endif
			delete cudaKernels;
		} else if (algo==3) {
			cout << endl << " *** Installing Subscriptions *** " << endl;
			CudaKernelsBloom *cudaKernels = new CudaKernelsBloom();
			t->start();
			for (map<int, set<CudaFilter *> >::iterator it=cudaSubscriptions.begin(); it!=cudaSubscriptions.end(); ++it) {
				cudaKernels->ifConfig(it->first, it->second);
			}
			cudaKernels->consolidate();
			double installTime = t->stop();

			cout << endl << " *** Deleting subscriptions *** " << endl;
			for (map<int, set<simple_filter *> >::iterator it=subscriptions.begin(); it!=subscriptions.end(); ++it) {
				for (set<simple_filter *>::iterator it2=it->second.begin(); it2!=it->second.end(); ++it2) {
					simple_filter *filter = *it2;
					delete filter;
				}
			}

			int numMessages = 0;
			int numResults = 0;
			double time = 0;
			cout << endl << " *** Running Test *** " << endl;
/*int dev=0;
			for (set<CudaMessage *>::iterator it=cudaMessages.begin(); it!=cudaMessages.end(); ++it) {
				CudaOutbox *outbox = new CudaOutbox;
				outbox->message = *it;
				t->start();
				cudaKernels->processMessage(outbox,dev);
				cudaKernels->getMatchingInterfaces(outbox->outgoingInterfaces,dev%ngpus);
				time += t->stop();
				numResults += outbox->outgoingInterfaces.size();
				numMessages++;
                dev++;
				delete outbox;
			}*//*
			double avgTime = time/numMessages;
			writeToFile(s, filename, label, numResults, installTime, avgTime);
			cout << " *** Test Executed *** " << endl;
			cout << " *** Generated Results: " << numResults << " *** " << endl;
			cout << " *** Processing Time: " << avgTime << " *** " << endl;
#if STATS==1
			double hToD, exec, dToH;
			cudaKernels->getStats(hToD, exec, dToH);
			cout << " *** Copy Data / Processing / Copy Results: " << hToD << " / " << exec << " / " << dToH << " *** " << endl;
			cout << endl << " *** Deleting table *** " << endl;
#endif
			delete cudaKernels;
		} else if (algo==4) {
			cout << endl << " *** Installing Subscriptions *** " << endl;
			CudaKernelsSimpleBloom *cudaKernels = new CudaKernelsSimpleBloom();
			t->start();
			for (map<int, set<CudaFilter *> >::iterator it=cudaSubscriptions.begin(); it!=cudaSubscriptions.end(); ++it) {
				cudaKernels->ifConfig(it->first, it->second);
			}
			cudaKernels->consolidate();
			double installTime = t->stop();

			cout << endl << " *** Deleting subscriptions *** " << endl;
			for (map<int, set<simple_filter *> >::iterator it=subscriptions.begin(); it!=subscriptions.end(); ++it) {
				for (set<simple_filter *>::iterator it2=it->second.begin(); it2!=it->second.end(); ++it2) {
					simple_filter *filter = *it2;
					delete filter;
				}
			}

			int numMessages = 0;
			int numResults = 0;
			double time = 0;
			cout << endl << " *** Running Test *** " << endl;
			for (set<CudaMessage *>::iterator it=cudaMessages.begin(); it!=cudaMessages.end(); ++it) {
				CudaOutbox *outbox = new CudaOutbox;
				outbox->message = *it;
				t->start();
				cudaKernels->processMessage(outbox);
				time += t->stop();
				numResults += outbox->outgoingInterfaces.size();
				numMessages++;
				delete outbox;
			}
			double avgTime = time/numMessages;
			writeToFile(s, filename, label, numResults, installTime, avgTime);
			cout << " *** Test Executed *** " << endl;
			cout << " *** Generated Results: " << numResults << " *** " << endl;
			cout << " *** Processing Time: " << avgTime << " *** " << endl;
#if STATS==1
			double hToD, exec, dToH;
			cudaKernels->getStats(hToD, exec, dToH);
			cout << " *** Copy Data / Processing / Copy Results: " << hToD << " / " << exec << " / " << dToH << " *** " << endl;
			cout << endl << " *** Deleting table *** " << endl;
#endif
			delete cudaKernels;
		}
		cout << endl << " *** Deleting messages *** " << endl;
		for (set<simple_message *>::iterator it=messages.begin(); it!=messages.end(); ++it) {
			simple_message *message = *it;
			delete message;
		}
		for (set<CudaMessage *>::iterator it=cudaMessages.begin(); it!=cudaMessages.end(); ++it) {
			CudaMessage *message = *it;
			delete message;
		}
		cout << endl << " *** Test finished *** " << endl;
	}*/
        }

}

void EvalRunner::writeToFile(int seed, string &filename, double label, int numResults, double install, double value) {
	string algo;
	stringstream stream;
	stream << seed;
	string stringSeed = stream.str();
	string name = "Results/" + filename + "_cudaBF_" + stringSeed;
	ofstream file;
	file.open(name.data(), ios::app);
	file << label << "\t" << value << "\t" << numResults << "\t" << install << endl;
	file.close();
}
