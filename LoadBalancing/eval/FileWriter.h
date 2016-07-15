#ifndef FILEWRITER_H_
#define FILEWRITER_H_

#include <fstream>
#include <sstream>
#include <set>
#include "WorkloadGenerator.h"
#include "../cuda/cudaTypes.h"
#include "../marshalling/Marshaller.h"
#include "../common/Timer.h"
#include "../common/Translator.h"
#include "../common/Consts.h"

class FileWriter {

public:
	/**
	 * Constructor
	 */
	FileWriter(WorkloadGenerator *workloadGenerator, Translator *translator);

	/**
	 * Destructor
	 */
	virtual ~FileWriter();

	/**
	 * Generates interfaces files
	 */
	void generateInterfaceFiles(int numInterfaces);

	/**
	 * Serializes the message contained in the give outbox and writes it to file.
	 * The message is written once for each interface matched.
	 */
	void writeResultsToFile(CudaOutbox *outbox);

private:
	Translator *translator;											// Translator
	WorkloadGenerator *workloadGenerator;				// WorkloadGenerator
	Marshaller *marshaller;											// Marshaller

};

#endif /* FILEWRITER_H_ */
