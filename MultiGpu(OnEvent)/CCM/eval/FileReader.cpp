#include "FileReader.h"

using namespace std;

FileReader::FileReader() {
	unmarshaller = new Unmarshaller();
}

FileReader::~FileReader() {
	delete unmarshaller;
}

CudaOutbox * FileReader::readFromFile(int interfaceNum, int &pos) {
	stringstream stream;
	stream << interfaceNum;
	string currentIfString = stream.str();
	string name = "Ramdisk/input" + currentIfString;
	ifstream file;
	file.open(name.data(), ios::in);
	file.seekg(pos, ios::beg);
	char sizeChar[4];
	file.read(sizeChar, 4);
	int size = unmarshaller->decodeInt(sizeChar);
	pos = pos + 4 + size;
	if (size<=0) {
		file.close();
		return NULL;
	}
	char charArray[size];
	file.read(charArray, size);
	CudaMessage *message = unmarshaller->decodeMessage(charArray);
	file.close();
	CudaOutbox *outbox = new CudaOutbox;
	outbox->message = message;
	return outbox;
}
