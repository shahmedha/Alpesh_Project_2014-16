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
  //runFiltersPerInterfaceScenario(algo);
  //runDefaultScenario(algo);
  //runAttributeScenario(algo);	 
  //runConstraintsPerFilterScenario(algo);
	runInterfacesScenario(algo);
/* runZipfScenario(algo);
  runConstraintsPerFilterFixedScenario(algo, 200000);
  runConstraintsPerFilterFixedScenario(algo, 1000000);
  runConstraintsPerFilterFixedScenario(algo, 5000000);
  runFiltersPerInterfaceScenario(algo);
  //runInterfacesFixedScenario(algo, 200000);
  //runInterfacesFixedScenario(algo, 1000000);
  //runInterfacesFixedScenario(algo, 5000000);
  runInterfacesScenario(algo);
  runNamesScenario(algo);
  //runTypeScenario(algo);
  runAttributeScenario(algo);
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
	for (int filter=100; filter<=250000; ) {
		int filterPerc = filter/10;
		paramHandler->setMinFilters(filter-filterPerc);
		paramHandler->setMaxFilters(filter+filterPerc);
		executeTest(algo, 0, 9, name, filter);
		if (filter<1000) filter+=300;
		else if (filter<10000) filter+=3000;
		else if (filter<100000) filter+=30000;
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
		executeTest(algo, 0, 9, name, i);
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
		executeTest(algo, 0, 9, name, i);
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

        ///Initialize Mpi Envronment 
         MPI_Init(NULL, NULL);
        
	  
          // Get the number of processes
          int world_size;
          MPI_Comm_size(MPI_COMM_WORLD, &world_size);

          // Get the rank of the process
          int world_rank;
          MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

          // Get the name of the processor
          char processor_name[MPI_MAX_PROCESSOR_NAME];
          int name_len;
          MPI_Get_processor_name(processor_name, &name_len);

          // Print off a hello world message
         //printf("Hello world from processor %s, rank %d out of %d processors\n", processor_name, world_rank, world_size);
        
        int device=0;
        
        if(world_rank==0){
                    int arr[5];
                    for(int i=0;i<5;i++)
                          arr[i]=i; 
                 cout<<"     ASDFSDF      ";
                std::ifstream infile(".mpi_hostfile");
                 string a, b;
                   int rank=1;
                  while (infile >> a >> b)
                {
                                
                                int number=0,flg=0;
                                for(int i=0;i<(int)b.size();i++){
                                        if(b[i]=='='){
                                                    flg=1;
                                   }
                                   else if(flg==1){
                                                number=number*10+b[i]-'0';
                                   }
                                }
                                        if(rank==1)
                                                number--;
                                for(int i=0;i<number;i++){
                                                arr[i]=i;
                                                        cout<<" Send from 0 to "<<rank<<"\n";
                                                MPI_Send(&arr[i],1,MPI_INT,rank,0, MPI_COMM_WORLD);
                                                rank++;
                                }
                                cout<<a<<" "<<b<<" "<<number<<endl;
                                
                }
         }
         else{
                        
                MPI_Recv(&device,1,MPI_INT,0,0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                cout<<" Recv from 0 to "<<device<<"\n"; 
         }
         //printf("  --Hello world from processor %s, rank %d out of %d processors got device %d\n", processor_name, world_rank, world_size,device);
	for (int s=minSeed; s<=maxSeed; s++) {
		//cout << endl << " #### Running test " << filename << " " << label << " seed=" << s << " #### " << endl;
		srand(s);
		set<simple_message *> messages;
		map<int, set<simple_filter *> > subscriptions;
		set<CudaMessage *> cudaMessages;
		map<int, set<CudaFilter *> > cudaSubscriptions;

		
		
		workloadGenerator->generateMessages(messages);

		
		workloadGenerator->generateSubscriptions(subscriptions);
              
		
		 
		translator->translateMessages(messages, cudaMessages);

		
		translator->translatePredicate(subscriptions, cudaSubscriptions);

		if (algo==1) {
			
			CudaKernels *cudaKernels = new CudaKernels(device);
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
			double time = 0,temp=0;
			cout << endl << " *** Running Test *** " << endl;
			int ind=1;
			int numofmes=0;
			
			    //    freopen("out.txt","w",stderr);
			 MPI_Request mpirequest;
			 MPI_Status status;
			 if(world_rank==0){
			        t->start();
			        set<CudaMessage *>::iterator it=cudaMessages.begin();
			        int cnt=0;
			
			        while(it!=cudaMessages.end()) {
					 MPI_Recv(&numofmes,1,MPI_INT,MPI_ANY_SOURCE,0, MPI_COMM_WORLD,&status);              
			                //cout<<" tt "<<t->stop();
			               	// cerr<<" Receive Source  "<<status.MPI_SOURCE<<" "<<numofmes<<" "<<cnt++<<endl;
			              //t->start();
			              MPI_Isend((void*)*it, sizeof(CudaMessage), MPI_BYTE, status.MPI_SOURCE, 0, MPI_COMM_WORLD,&mpirequest);
			                it++;
			                   
			               
			               numResults+=numofmes;
			               numMessages++;
			        }
			        int ccn=1;
			        while(ccn<world_size){
			                		MPI_Recv(&numofmes,1,MPI_INT,ccn++,0, MPI_COMM_WORLD,&status);  
			                		numResults+=numofmes;            
			        }
			        time+=t->stop();
			        /// probe all other process to stop the execution 
			        CudaMessage *probe  = new CudaMessage;
			        probe->numAttributes=0;
			        for(int i=1;i<world_size;i++){
			                MPI_Send((void*)probe, sizeof(CudaMessage), MPI_BYTE, (i) % world_size, 0, MPI_COMM_WORLD);
			        }
			}
			else{
			               MPI_Send(&numofmes,1,MPI_INT,0,0, MPI_COMM_WORLD);
			               while(true){
			               CudaMessage *it = new CudaMessage;
			               MPI_Recv((void*)it, sizeof(CudaMessage),MPI_BYTE, 0 , 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 	
			               ///break condition all the messages are finished executing
			                 if(it->numAttributes==0){
			                        cout<<" Out of Loop "<<world_rank<<endl;           
			                        break;
			               }
			              numMessages++;
				        CudaOutbox *outbox = new CudaOutbox;
				        outbox->message = it;
				        t->start();
				        cudaKernels->processMessage(outbox);
				        time = t->stop();
				        numResults += outbox->outgoingInterfaces.size();
									        
					numofmes=outbox->outgoingInterfaces.size();
                                               
				        MPI_Send(&numofmes,1,MPI_INT,0,0, MPI_COMM_WORLD);
				        
				          delete outbox;
				          }
		        }
		       
		   
		         //MPI_Barrier(MPI_COMM_WORLD);
		         double hToD, exec, dToH;
			cudaKernels->getStats(hToD, exec, dToH);
		        if(world_rank==0){
			cout<<" Time Required for 1000 mess "<<time<<endl;
			double avgTime = time/numMessages;
			writeToFile(s, filename, label, numResults, installTime, avgTime);
			cout << " *** Test Executed *** " << endl;
			cout << " *** Generated Results: " << numResults << " *** " << endl;
			cout << " *** Processing Time: " << avgTime << " *** " << endl;
#if STATS==1
			
			
#endif
                         }
                         else if(world_rank==1){
                         cout << " *** Copy Data / Processing / Copy Results: " << hToD << " / " << exec << " / " << dToH << " *** " << endl;
			        cout << endl << " *** Deleting table *** " << endl;
			}
			delete cudaKernels;
		}
		}
	cout<<"Hello Everyone "<<endl; 
	MPI_Abort(MPI_COMM_WORLD,0);
	 MPI_Finalize();
	 

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
/*/**
else if (algo==2) {
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


