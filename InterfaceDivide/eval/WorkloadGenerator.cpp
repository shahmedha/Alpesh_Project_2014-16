#include "WorkloadGenerator.h"

WorkloadGenerator::WorkloadGenerator(ParamHandler *paramHandler) {
	this->paramHandler = paramHandler;
	computeNameSet();
	computeStringValues();
}

WorkloadGenerator::~WorkloadGenerator() {
	// Nothing to do
}

void WorkloadGenerator::generateMessages(set<simple_message *> &messages) {
	int numMessages = paramHandler->getNumMessages();
	for (int i=0; i<numMessages; i++) {
		generateAndAddMessage(messages);
	}
}

void WorkloadGenerator::generateSubscriptions(map<int, set<simple_filter *> > &subscriptions) {
	int numInterfaces = paramHandler->getNumIf();
	for (int i=0; i<numInterfaces; i++) {
		int numFilters = paramHandler->getMinFilters()+rand()%(1+paramHandler->getMaxFilters()-paramHandler->getMinFilters());
		set<simple_filter *> filtersSet;
		for (int f=0; f<numFilters; f++) {
			generateAndAddFilter(filtersSet);
		}
		subscriptions.insert(make_pair(i, filtersSet));
	}
}

void WorkloadGenerator::resetNames() {
	names.clear();
	computeNameSet();
}

void WorkloadGenerator::resetStringValues() {
	stringValues.clear();
	computeStringValues();
}

int WorkloadGenerator::getZipf(double alpha, int n) {
	static bool first = true;
	static double c = 0;
	double z;
	double sum_prob;
	int result = 0;

	// Compute normalization constant on first call only
	if (first==true) {
		for (int i=1; i<=n; i++) {
			c = c + (1.0 / pow((double) i, alpha));
		}
		c = 1.0 / c;
		first = false;
	}

	// Pull a uniform random number (0 < z < 1)
	do {
		z = (double) rand()/RAND_MAX;
	} while ((z == 0) || (z == 1));

	// Map z to the value
	sum_prob = 0;
	for (int i=1; i<=n; i++) {
		sum_prob = sum_prob + c / pow((double) i, alpha);
		if (sum_prob >= z) {
			result = i;
			break;
		}
	}
	return result;
}

void WorkloadGenerator::generateAndAddFilter(set<simple_filter *> &subscriptions) {
	simple_filter *filter = new simple_filter();
	int numConstr = paramHandler->getMinConstr()+rand()%(1+paramHandler->getMaxConstr()-paramHandler->getMinConstr());
	set<int> alreadyConsidered;
	for (int i=0; i<numConstr; i++) {
	  addConstraint(filter, alreadyConsidered);
	}
	subscriptions.insert(filter);
}

void WorkloadGenerator::generateAndAddMessage(set<simple_message *> &messages) {
	simple_message *message = new simple_message();
	int numAttr = paramHandler->getMinAttr()+rand()%(1+paramHandler->getMaxAttr()-paramHandler->getMinAttr());
	for (int i=0; i<numAttr; i++) {
		addAttribute(message);
	}
	messages.insert(message);
}

void WorkloadGenerator::addAttribute(simple_message *message) {
	int nameRand = 0;
	if (paramHandler->getZipfNames()) nameRand = getZipf(1, paramHandler->getNumNames())-1;
	else nameRand = rand()%paramHandler->getNumNames();
	siena::type_id type = getType(nameRand);
	siena::string_t name = names[nameRand];
	simple_value *simpleValue;
	if (type==siena::int_id) {
		siena::int_t intVal = siena::int_t(rand()%paramHandler->getNumAttVal());
		simpleValue = new simple_value(intVal);
	} else {
		int stringValRand = rand()%paramHandler->getNumAttVal();
		siena::string_t stringVal = stringValues[stringValRand];
		simpleValue = new simple_value(stringVal);
	}
	message->add(name, simpleValue);
}

void WorkloadGenerator::addConstraint(simple_filter *filter, set<int> &alreadyConsidered) {
  int nameRand = 0;
  bool isNew;
  do {
    if (paramHandler->getZipfNames()) nameRand = getZipf(1, paramHandler->getNumNames())-1;
    else nameRand = rand()%paramHandler->getNumNames();
    // check if this name has already been used in this filter
    if (alreadyConsidered.find(nameRand)==alreadyConsidered.end()) isNew=true;
    else isNew=false;
  } while(!isNew);
  alreadyConsidered.insert(nameRand);
  siena::type_id type = getType(nameRand);
  siena::operator_id opId = getOp(type);
  siena::string_t name = names[nameRand];
  simple_op_value *opValue;
  if (type==siena::int_id) {
    siena::int_t intVal = rand()%paramHandler->getNumAttVal();
    opValue = new simple_op_value(opId, intVal);
  } else {
    int stringValRand = rand()%paramHandler->getNumAttVal();
    siena::string_t stringVal = stringValues[stringValRand];
    opValue = new simple_op_value(opId, stringVal);
  }
  filter->add(name, opValue);
}

void WorkloadGenerator::computeNameSet() {
	for (int i=0; i<paramHandler->getNumNames(); i++) {
		char *name = new char[NAME_LEN];
		for (int j=0; j<NAME_LEN-1; j++) {
			name[j] = 'a'+rand()%25;
		}
		name[NAME_LEN-1] = '\0';
		siena::string_t str(name);
		names.push_back(str);
		nameIds.insert(make_pair(str,names.size()-1));
	}
}

void WorkloadGenerator::computeStringValues() {
	// If all names are referred to integer values, there is no need to compute string values
	if (paramHandler->getNumNames()==paramHandler->getNumIntNames()) return;
	for (int i=0; i<paramHandler->getNumAttVal(); i++) {
		char *name = new char[STRING_VAL_LEN];
		for (int j=0; j<STRING_VAL_LEN-1; j++) {
			name[j] = 'a'+rand()%25;
		}
		name[STRING_VAL_LEN-1] = '\0';
		siena::string_t str(name);
		stringValues.push_back(str);
	}
}

siena::type_id WorkloadGenerator::getType(int name) {
	if (name<paramHandler->getNumIntNames()) return siena::int_id;
	else return siena::string_id;
}

siena::operator_id WorkloadGenerator::getOp(siena::type_id type) {
	double r = rand()%100;
	double count = 0;
	if (type==siena::int_id) {
		count += paramHandler->getPercIntEq();
		if (r<count) return siena::eq_id;
		count += paramHandler->getPercIntDf();
		if (r<count) return siena::ne_id;
		count += paramHandler->getPercIntLt();
		if (r<count) return siena::lt_id;
		count += paramHandler->getPercIntGt();
		return siena::gt_id;
	} else {
		count += paramHandler->getPercStrEq();
		if (r<count) return siena::eq_id;
		count += paramHandler->getPercStrDf();
		if (r<count) return siena::ne_id;
		count += paramHandler->getPercStrIn();
		if (r<count) return siena::ss_id;
		count += paramHandler->getPercStrPf();
		return siena::pf_id;
	}
}
