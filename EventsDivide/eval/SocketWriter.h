#ifndef SOCKETWRITER_H_
#define SOCKETWRITER_H_

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <fstream>
#include <sstream>
#include <set>
#include "WorkloadGenerator.h"
#include "../cuda/cudaTypes.h"
#include "../marshalling/Marshaller.h"
#include "../common/Translator.h"
#include "../common/Consts.h"
#include "../common/Timer.h"

class SocketWriter {

public:
	/**
	 * Constructor
	 */
	SocketWriter(int numMessages);

	/**
	 * Destructor
	 */
	virtual ~SocketWriter();

	/**
	 * Generates messages and stores them in main memory
	 */
	void generateMessages(int numMessages, int seed);

	/**
	 * Connect to server
	 */
	void connectToServer();

	/**
	 * Starts sending messages at the given rate
	 */
	void startSendingMessages(int microSleep);

private:
	ParamHandler *paramHandler;									// PArameter handler
	Translator *translator;											// Translator
	WorkloadGenerator *workloadGenerator;				// WorkloadGenerator
	Marshaller *marshaller;											// Marshaller
	std::set<CudaMessage *> cudaMessages;					// Messages
	int sock;																		// The socket to communicate with the server

};

#endif /* FILEWRITER_H_ */
