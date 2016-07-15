#ifndef TRANSLATOR_H_
#define TRANSLATOR_H_

#include "../sff/simple_fwd_types.h"
#include "../cuda/cudaTypes.h"
#include "../sff/siena/types.h"
#include <set>
#include <iostream>
#include <stdlib.h>

using namespace std;
using namespace siena;

class Translator {
public:

	Translator();

	virtual ~Translator();

	void translateMessages(set<simple_message *> &in, set<CudaMessage *> &out);
	void translatePredicate(map<int, set<simple_filter *> > &in, map<int, set<CudaFilter *> > &out);

private:

	void translateMessage(simple_message *in, CudaMessage *out);
	void translateOp(operator_id in, Op &out);
	void translateFilter(simple_filter *in, CudaFilter *out);

};

#endif /* TRANSLATOR_H_ */
