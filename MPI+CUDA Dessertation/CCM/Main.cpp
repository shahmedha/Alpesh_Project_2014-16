#include "common/Consts.h"
#include "eval/EvalRunner.h"
#include "eval/SystemEvalRunner.h"
#include "eval/SocketWriter.h"
#include <iostream>
#include <stdlib.h>

using namespace std;

void usage() {
  cout << "Usage: ./CCM [0-6]" << endl;
  cout << "0:    Starts client for throughput measure" << endl;
  cout << "          The number of submitted messages, the size of the input queue, and the network address of the server can be changed" << endl;
  cout << "          in common/Consts.h (needs recompiling)" << endl;
  cout << "1:    Run latency measures under different workloads. Standard protocol" << endl;
  cout << "2:    Run latency measures under different workloads. NoDup protocol" << endl;
  cout << "3:    Run latency measures under different workloads. Bloom enhanced protocol" << endl;
  cout << "4:    Run latency measures under different workloads. Simplified Bloom enhanced protocol" << endl;
  cout << "5:    Run throughput test reading events from local file (no need for an external client). Adopts the Simplified Bloom enhanced protocol" << endl;
  cout << "6:    Run throughput test reading events from network (requires an external client to send events). Adopts the Simplified Bloom enhanced protocol" << endl;
  exit(1);
}

int main(int argc, char *argv[]) {
  int algo = 0;
  if(argc!=2) usage();
  else if(argv[1][0]=='0') algo = 0;
  else if(argv[1][0]=='1') algo = 1;
  else if(argv[1][0]=='2') algo = 2;
  else if(argv[1][0]=='3') algo = 3;
  else if(argv[1][0]=='4') algo = 4;
  else if(argv[1][0]=='5') algo = 5;
  else if(argv[1][0]=='6') algo = 6;
  else usage();
  
  if(algo==0) {
    for (int i=1200; i>=20; ) {
      for (int run=0; run<10; run++) {
	SocketWriter *sockWriter = new SocketWriter(CLIENT_MESSAGES);
	sockWriter->generateMessages(CLIENT_MESSAGES, run);
	sockWriter->connectToServer();
	sockWriter->startSendingMessages(i);
	delete sockWriter;
	sleep(CLIENT_MESSAGES/(i*1000)+40);
      }
      if(i>100) i-=100;
      else i-=20;
    }
  } else if(algo>=1 && algo<=4) {
    EvalRunner *evalRunner = new EvalRunner();
    evalRunner->runTests(algo);
    delete evalRunner;
  } else if(algo>=5 && algo<=6) {
    SystemEvalRunner *evalRunner = new SystemEvalRunner(algo, QUEUE_SIZE, CLIENT_MESSAGES);
    evalRunner->runTests();
    delete evalRunner;
  } else {
    cout << "Error: invalid algorithm number" << endl;
  }
  return 0;
}

