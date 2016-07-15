#ifndef MARSHALLER_H_
#define MARSHALLER_H_

#include <iostream>
#include <string.h>
#include "../common/Consts.h"
#include "../cuda/cudaTypes.h"

/**
 * This file contains all the functions used to encode packets as array of bytes.
 */
class Marshaller {

public:
	void encodeEnd(char *dest);
	void encode(CudaMessage *source, char *dest);
	int getNumBytes(CudaMessage *source);

private:
	void encode(CudaMessage *source, char *dest, int &index);

	int getNumBytes(CudaAttribute &source);
	void encode(CudaAttribute &source, char *dest, int &index);

	int getNumBytes(int source);
	void encode(int source, char *dest, int &index);

	int getNumBytes(char *source);
	void encode(char *source, char *dest, int &index);

	int getNumBytes(CudaValue &source);
	void encode(CudaValue &source, char *dest, int &index);

	int getNumBytes(Type source);
	void encode(Type source, char *dest, int &index);

};

#endif /* MARSHALLER_H_ */
