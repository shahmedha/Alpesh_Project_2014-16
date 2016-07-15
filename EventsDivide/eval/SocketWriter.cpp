#include "SocketWriter.h"

SocketWriter::SocketWriter(int numMessages) {
	paramHandler = new ParamHandler();
	paramHandler->setNumMessages(numMessages);
	marshaller = new Marshaller();
	translator = new Translator();
	workloadGenerator = new WorkloadGenerator(paramHandler);
}

SocketWriter::~SocketWriter() {
	delete paramHandler;
	delete marshaller;
	delete translator;
	delete workloadGenerator;
}

void SocketWriter::generateMessages(int numMessages, int seed) {
	srand(seed);
	set<simple_message *> messages;
	cout << endl << " *** Generating messages *** " << endl;
	workloadGenerator->resetNames();
	workloadGenerator->resetStringValues();
	workloadGenerator->generateMessages(messages);
	cout << endl << " *** Translating messages *** " << endl;
	translator->translateMessages(messages, cudaMessages);
	cout << endl << " *** Deleting siena messages *** " << endl;
	for (set<simple_message *>::iterator it=messages.begin(); it!=messages.end(); ++it) {
		simple_message *message = *it;
		delete message;
	}
}

void SocketWriter::connectToServer() {
	struct sockaddr_in sa;
	memset(&sa, 0, sizeof(struct sockaddr_in));
	sa.sin_family = AF_INET;
	sa.sin_port = htons(9000);
	sa.sin_addr.s_addr = inet_addr(SERVER_ADDR);

	sock = socket(AF_INET, SOCK_STREAM, 0);
	if (socket<0) {
		cout << "Error creating the socket!" << endl;
		exit(-1);
	}
	if (connect(sock, (struct sockaddr *) &sa, sizeof(sa))<0) {
		cout << "Error while connecting!" << endl;
		exit(-1);
	} else {
		cout << " *** Connection established with the server *** " << endl;
	}
}

void SocketWriter::startSendingMessages(int microSleep) {
	Timer t;
	cout << endl <<  " *** Start sending messages *** " << endl;
	for (set<CudaMessage *>::iterator it=cudaMessages.begin(); it!=cudaMessages.end(); ++it) {
		t.start();
		CudaMessage *message = *it;
		int size = marshaller->getNumBytes(message);
		char *serializedMessage = new char[size];
		marshaller->encode(message, serializedMessage);
		write(sock, serializedMessage, size);
		delete [] serializedMessage;
		delete message;
		while (t.stop()<=microSleep/1000.0) { }
	}
	cout << endl << " *** All messages have been sent - Closing socket and terminating *** " << endl;
	close(sock);
}
