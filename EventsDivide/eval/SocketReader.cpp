#include "SocketReader.h"

SocketReader::SocketReader() {
	unmarshaller = new Unmarshaller();
}

SocketReader::~SocketReader() {
	delete unmarshaller;
}

void SocketReader::bindAddress(int port) {
	struct sockaddr_in sa;
	memset(&sa, 0, sizeof(struct sockaddr_in));
	sa.sin_family = AF_INET;
	sa.sin_port = htons(port);
	sa.sin_addr.s_addr = htonl(INADDR_ANY);
	// Create the socket and bind it
	serverSocket = socket (AF_INET, SOCK_STREAM, 0);
	if (serverSocket<0) {
		perror("creating the socket");
		exit(-1);
	}
	if (bind(serverSocket, (struct sockaddr *)&sa, sizeof(sa))<0){
		perror("binding");
		exit(-1);
	}
}

void SocketReader::openConnection() {
	if (listen(serverSocket, 0)<0) {
		perror("listening");
		exit(-1);
	}
	// Accept a new connection
	clientSocket = accept(serverSocket, NULL, NULL);
	if (clientSocket<0) {
		perror("accepting");
		exit(-1);
	}
}

void SocketReader::closeServerSocket() {
	close(serverSocket);
}

CudaOutbox * SocketReader::readFromNetwork() {
	// Reads the packet length
	char lengthByteArray[4];
	int n = 0;
	int alreadyRead = 0;
	while (alreadyRead < 4) {
		n = read(clientSocket, lengthByteArray+alreadyRead, 4-alreadyRead);
		if (n<=0) {
			close(clientSocket);
			return NULL;
		}
		alreadyRead += n;
	}
	int length = unmarshaller->decodeInt(lengthByteArray);
	if (length<=0) {
		close(clientSocket);
		return NULL;
	}
	// Reads the packet
	char pktByteArray[length];
	alreadyRead = 0;
	while (alreadyRead < length) {
		n = read(clientSocket, pktByteArray+alreadyRead, length-alreadyRead);
		if (n<=0) {
			close(clientSocket);
			return NULL;
		}
		alreadyRead += n;
	}
	CudaMessage *message = unmarshaller->decodeMessage(pktByteArray);
	CudaOutbox *outbox = new CudaOutbox;
	outbox->message = message;
	return outbox;
}
