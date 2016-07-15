#ifndef UNMARSHALLER_H_
#define UNMARSHALLER_H_

#include <string.h>
#include "../common/Consts.h"
#include "../cuda/cudaTypes.h"

/**
 * This file contains all the functions used to decode packets starting from an array of bytes.
 */
class Unmarshaller {

public:
	CudaMessage * decodeMessage(char *source);
	int decodeInt(char *source);

private:
	int decodeInt(char *source, int &index);
	void decodeMessage(CudaMessage *dest, char *source, int &index);
	void decodeAttribute(CudaAttribute &dest, char *source, int &index);
	void decodeString(char *dest, char *source, int &index);
	void decodeCudaValue(CudaValue &dest, char *source, int &index);
	void decodeType(Type &dest, char *source, int &index);

};

#endif /* UNMARSHALLER_H_ */
