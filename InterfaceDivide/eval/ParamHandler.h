#ifndef PARAMHANDLER_H_
#define PARAMHANDLER_H_

class ParamHandler {
public:

	/**
	 * Constructor
	 */
	ParamHandler();

	/**
	 * Destructor
	 */
	virtual ~ParamHandler();

	/**
	 * Reset to default values
	 */
	void resetToDefault();

	/**
	 * Getters and setters
	 */
	int getNumMessages() const;
	int getMaxAttr() const;
	int getMaxConstr() const;
	int getMaxFilters() const;
	int getMinAttr() const;
	int getMinConstr() const;
	int getMinFilters() const;
	int getNumAttVal() const;
	int getNumIf() const;
	int getNumIntNames() const;
	int getNumNames() const;
	int getConstraintsPerThread() const;
	void setNumMessages(int numMessages);
	void setMaxAttr(int maxAttr);
	void setMaxConstr(int maxConstr);
	void setMaxFilters(int maxFilters);
	void setMinAttr(int minAttr);
	void setMinConstr(int minConstr);
	void setMinFilters(int minFilters);
	void setNumAttVal(int numAttVal);
	void setNumIf(int numIf);
	void setNumIntNames(int numIntNames);
	void setNumNames(int numNames);
	void setConstraintsPerThread(int constraintsPerThread);
	int getPercIntDf();
	int getPercIntEq();
	int getPercIntGt();
	int getPercIntLt();
	int getPercStrDf();
	int getPercStrEq();
	int getPercStrIn();
	int getPercStrPf();
	bool getZipfNames();
	void setPercIntDf(int percIntDf);
	void setPercIntEq(int percIntEq);
	void setPercIntGt(int percIntGt);
	void setPercIntLt(int percIntLt);
	void setPercStrDf(int percStrDf);
	void setPercStrEq(int percStrEq);
	void setPercStrIn(int percStrIn);
	void setPercStrPf(int percStrPf);
	void setZipfNames(bool zipfNames);

private:
	int numMessages;					// Number of messages
	int numIf;								// Number of interfaces
	int minFilters;						// Minimum number of filter for each interface
	int maxFilters;						// Maximum number of filter for each interface
	int minConstr;						// Minimum number of constraints for each filter
	int maxConstr;						// Maximum number of constraints for each filter
	int minAttr;							// Minimum number of attributes for each message
	int maxAttr;							// Maximum number of attributes for each message
	int numAttVal;						// Number of values for each attribute
	int numNames;							// Number of names for attributes
	int numIntNames;					// Number of names for int attributes
	int percIntEq;						// Percentage of eq operator in int constraints
	int percIntDf;						// Percentage of df operator in int constraints
	int percIntGt;						// Percentage of gt operator in int constraints
	int percIntLt;						// Percentage of lt operator in int constraints
	int percStrEq;						// Percentage of eq operator in string constraints
	int percStrDf;						// Percentage of df operator in string constraints
	int percStrIn;						// Percentage of in operator in string constraints
	int percStrPf;						// Percentage of pf operator in string constraints
	bool zipfNames;						// True if names are selected from a zipf distribution
	int contraintsPerThread;	// Constraints to be processed by a single thread

	inline void setDefaultValues();

};

#endif /* PARAMHANDLER_H_ */
