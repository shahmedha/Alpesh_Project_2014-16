#ifndef WORKLOADGENERATOR_H_
#define WORKLOADGENERATOR_H_

#include "ParamHandler.h"
#include <iostream>
#include <set>
#include <map>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include "../sff/simple_fwd_types.h"
#include "../common/Consts.h"

using namespace std;

class WorkloadGenerator {
public:

	WorkloadGenerator(ParamHandler *paramHandler);

	virtual ~WorkloadGenerator();

	/**
	 * Generates messages according to the information stored in the paramHandler
	 */
	void generateMessages(set<simple_message *> &messages);

	/**
	 * Generate subscriptions according to the information stored in the paramHandler
	 */
	void generateSubscriptions(map<int, set<simple_filter *> > &subscriptions);

	/**
	 * Computes again the set of names
	 */
	void resetNames();

	/**
	 * Computes again the set of values for string attributes
	 */
	void resetStringValues();

	/**
	 * Returns a random number with the zipf distribution
	 */
	int getZipf(double alpha, int n);

	map<siena::string_t, int> nameIds;      // map names to name identifiers (numbers)

private:
	ParamHandler *paramHandler;									// Param handler
	vector<siena::string_t> names;							// Set of all possible names
	vector<siena::string_t> stringValues;				// Set of all possible values for strings

	void generateAndAddFilter(set<simple_filter *> &subscriptions);
	void generateAndAddMessage(set<simple_message *> &messages);
	void addAttribute(simple_message *message);
	void addConstraint(simple_filter *filter, std::set<int> &alreadyConsidered);

	void computeNameSet();
	void computeStringValues();

	siena::type_id getType(int name);
	siena::operator_id getOp(siena::type_id type);

};

#endif /* WORKLOADGENERATOR_H_ */
