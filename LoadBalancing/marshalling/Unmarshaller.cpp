#include "Unmarshaller.h"

CudaMessage * Unmarshaller::decodeMessage(char *source) {
	CudaMessage *message = new CudaMessage;
	int index = 0;
	decodeMessage(message, source, index);
	return message;
}

int Unmarshaller::decodeInt(char *source) {
	int index = 0;
	return decodeInt(source, index);
}

int Unmarshaller::decodeInt(char *source, int &index) {
	int result = (0xff & source[index+0]) << 24 |
							 (0xff & source[index+1]) << 16 |
							 (0xff & source[index+2]) << 8  |
							 (0xff & source[index+3]) << 0;
	index += 4;
	return result;
}

void Unmarshaller::decodeMessage(CudaMessage *dest, char *source, int &index) {
	dest->numAttributes = decodeInt(source, index);
	for (int i=0; i<dest->numAttributes; i++) {
		decodeAttribute(dest->attributes[i], source, index);
	}
}

void Unmarshaller::decodeAttribute(CudaAttribute &dest, char *source, int &index) {
	decodeString(dest.name, source, index);
	decodeCudaValue(dest.value, source, index);
}

void Unmarshaller::decodeString(char *dest, char *source, int &index) {
	int stringSize = decodeInt(source, index);
	for (int i=0; i<stringSize; i++) {
		dest[i] = source[index];
		index++;
	}
	dest[stringSize] = '\0';
}

void Unmarshaller::decodeCudaValue(CudaValue &dest, char *source, int &index) {
	decodeType(dest.type, source, index);
	if (dest.type==INT) dest.intVal = decodeInt(source, index);
	else decodeString(dest.stringVal, source, index);
}

void Unmarshaller::decodeType(Type &dest, char *source, int &index) {
	if (source[index]==0) dest = INT;
	else dest = STRING;
	index++;
}
