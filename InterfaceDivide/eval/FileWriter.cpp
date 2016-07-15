#include "FileWriter.h"

FileWriter::FileWriter(WorkloadGenerator *workloadGenerator, Translator *translator) {
	marshaller = new Marshaller();
	this->translator = translator;
	this->workloadGenerator = workloadGenerator;
}

FileWriter::~FileWriter() {
	delete marshaller;
}

void FileWriter::generateInterfaceFiles(int numInterfaces) {
	set<simple_message *> messages;
	set<CudaMessage *> cudaMessages;

	workloadGenerator->generateMessages(messages);
	translator->translateMessages(messages, cudaMessages);
	for (set<simple_message *>::iterator it=messages.begin(); it!=messages.end(); ++it) {
		simple_message *message = *it;
		delete message;
	}

	int numMessagesPerInterface = cudaMessages.size()/numInterfaces;
	int count = 0;
	int currentInterface = 0;
	for (set<CudaMessage *>::iterator it=cudaMessages.begin(); it!=cudaMessages.end(); ++it) {
		stringstream stream;
		stream << currentInterface;
		string currentIfString = stream.str();
		string name = "Ramdisk/input" + currentIfString;
		ofstream file;
		if (count==0) file.open(name.data(), ios::out);
		else file.open(name.data(), ios::app);
		CudaMessage *message = *it;
		int size = marshaller->getNumBytes(message);
		char *serializedMessage = new char[size];
		marshaller->encode(message, serializedMessage);
		file.write(serializedMessage, size);
		delete [] serializedMessage;
		count++;
		if (count==numMessagesPerInterface) {
			count = 0;
			currentInterface++;
			char *end = new char[4];
			marshaller->encodeEnd(end);
			file.write(end, 4);
		}
		file.close();
	}

	for (set<CudaMessage *>::iterator it=cudaMessages.begin(); it!=cudaMessages.end(); ++it) {
		CudaMessage *message = *it;
		delete message;
	}
}

void FileWriter::writeResultsToFile(CudaOutbox *outbox) {
	int size = marshaller->getNumBytes(outbox->message);
	char *serializedMessage = new char[size];
	marshaller->encode(outbox->message, serializedMessage);
	for (set<int>::iterator it=outbox->outgoingInterfaces.begin(); it!=outbox->outgoingInterfaces.end(); ++it) {
		int interface = *it;
		stringstream stream;
		stream << interface;
		string currentIfString = stream.str();
		string name = "Ramdisk/output" + currentIfString;
		ofstream file;
		file.open(name.data(), ios::app);
		file.write(serializedMessage, size);
		file.close();
	}
	delete [] serializedMessage;
}
