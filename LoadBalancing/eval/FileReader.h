#ifndef FILEREADER_H_
#define FILEREADER_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include "../cuda/cudaTypes.h"
#include "../marshalling/Unmarshaller.h"

class FileReader {

public:
	/**
	 * Constructor
	 */
	FileReader();

	/**
	 * Destructor
	 */
	virtual ~FileReader();

	/**
	 * Extracts a message from the file representing the interface with the given id
	 * Starts reading from the byte in position pos, and updates it.
	 * Return NULL if the file is empty
	 */
	CudaOutbox * readFromFile(int interfaceNum, int &pos);

private:
	Unmarshaller *unmarshaller;									// Unmarshaller

};

#endif /* FILEREADER_H_ */
