#include "Marshaller.h"

using namespace std;

void Marshaller::encodeEnd(char *dest) {
	int index = 0;
	int content = 0;
	encode(content, dest, index);
}

void Marshaller::encode(CudaMessage *source, char *dest) {
	int index = 0;
	int size = getNumBytes(source)-4;
	encode(size, dest, index);
	encode(source, dest, index);
}

int Marshaller::getNumBytes(CudaMessage *source) {
	int size = getNumBytes(source->numAttributes);
	for (int i=0; i<source->numAttributes; i++) {
		size += getNumBytes(source->attributes[i]);
	}
	return size+4;
}

void Marshaller::encode(CudaMessage *source, char *dest, int &index) {
	encode(source->numAttributes, dest, index);
	for (int i=0; i<source->numAttributes; i++) {
		encode(source->attributes[i], dest, index);
	}
}

int Marshaller::getNumBytes(CudaAttribute &source) {
	return getNumBytes(source.name) + getNumBytes(source.value);
}

void Marshaller::encode(CudaAttribute &source, char *dest, int &index) {
	encode(source.name, dest, index);
	encode(source.value, dest, index);
}

int Marshaller::getNumBytes(int source) {
	return 4;
}

void Marshaller::encode(int source, char *dest, int &index) {
	dest[index+0] = ((source >> 24) & 0xff);
	dest[index+1] = ((source >> 16) & 0xff);
	dest[index+2] = ((source >> 8) & 0xff);
	dest[index+3] = ((source >> 0) & 0xff);
	index += 4;
}

int Marshaller::getNumBytes(char *source) {
	int size = strlen(source);
	size += getNumBytes(size);
	return size;
}

void Marshaller::encode(char *source, char *dest, int &index) {
	int size = strlen(source);
	encode(size, dest, index);
	for (int i=0; i<size; i++) {
		dest[index] = source[i];
		index++;
	}
}

int Marshaller::getNumBytes(CudaValue &source) {
	if (source.type==INT) return getNumBytes(source.type) + getNumBytes(source.intVal);
	else return getNumBytes(source.type) + getNumBytes(source.stringVal);
}

void Marshaller::encode(CudaValue &source, char *dest, int &index) {
	encode(source.type, dest, index);
	if (source.type==INT) encode(source.intVal, dest, index);
	else encode(source.stringVal, dest, index);
}

int Marshaller::getNumBytes(Type source) {
	return 1;
}

void Marshaller::encode(Type source, char *dest, int &index) {
	if (source==INT) dest[index] = 0;
	else dest[index] = 1;
	index++;
}
