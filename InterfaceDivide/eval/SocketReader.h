#ifndef SOCKETREADER_H_
#define SOCKETREADER_H_

#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "../cuda/cudaTypes.h"
#include "../marshalling/Unmarshaller.h"

class SocketReader {

public:
	/**
	 * Constructor
	 */
	SocketReader();

	/**
	 * Destructor
	 */
	virtual ~SocketReader();

	/**
	 * Binds the address to accept connections on the given port
	 */
	void bindAddress(int port);

	/**
	 * Opens a connection with a publisher on the given port
	 */
	void openConnection();

	/**
	 * Closes the server socket
	 */
	void closeServerSocket();

	/**
	 * Reads a new message from the network and deserializes it.
	 * Returns NULL if the connection closes
	 */
	CudaOutbox * readFromNetwork();

private:
	Unmarshaller *unmarshaller;				// Unmarshaller
	int clientSocket;									// The socket for the connection with the client
	int serverSocket;									// The socket used to accept new connections

};

#endif /* SOCKETREADER_H_ */
