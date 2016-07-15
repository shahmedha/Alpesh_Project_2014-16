#include "Translator.h"

Translator::Translator() {
	// Nothing to do
}

Translator::~Translator() {
	// Nothing to do
}

void Translator::translateMessages(set<simple_message *> &in, set<CudaMessage *> &out) {
	for (set<simple_message *>::iterator it=in.begin(); it!=in.end(); ++it) {
		simple_message *message = *it;
		CudaMessage *cudaMessage = new CudaMessage();
		translateMessage(message, cudaMessage);
		out.insert(cudaMessage);
	}
}

void Translator::translatePredicate(map<int, set<simple_filter *> > &in, map<int, set<CudaFilter *> > &out) {
	for (map<int, set<simple_filter *> >::iterator it=in.begin(); it!=in.end(); ++it) {
		int interfaceId = it->first;
		set<CudaFilter *> cudaFiltersSet;
		for (set<simple_filter *>::iterator it2=it->second.begin(); it2!=it->second.end(); ++it2) {
			simple_filter *filter = *it2;
			CudaFilter *cudaFilter = new CudaFilter();
			translateFilter(filter, cudaFilter);
			cudaFilter->interface = interfaceId;
			cudaFiltersSet.insert(cudaFilter);
		}
		out.insert(make_pair(interfaceId, cudaFiltersSet));
	}
}

void Translator::translateMessage(simple_message *in, CudaMessage *out) {
	simple_message::iterator *it = in->first();
	out->numAttributes=0;
	do {
		if (out->numAttributes>MAX_ATTR_NUM) {
			cerr << "Overcoming the maximum number of constraints" << endl;
			exit(1);
		}
		for (unsigned int i=0; i<it->name().length(); i++) {
			out->attributes[out->numAttributes].name[i] = it->name()[i];
		}
		if (it->type()==int_id) {
			out->attributes[out->numAttributes].value.type = INT;
			out->attributes[out->numAttributes].value.intVal = it->int_value();
		} else if (it->type()==string_id) {
			out->attributes[out->numAttributes].value.type = STRING;
			for (unsigned int i=0; i<it->string_value().length(); i++) {
				out->attributes[out->numAttributes].value.stringVal[i] = it->string_value()[i];
			}
		} else {
			cerr << it->type() << "Unsupported value type" << endl;
			exit(1);
		}
		out->numAttributes++;
	} while(it->next());
	delete it;
}

void Translator::translateOp(operator_id in, Op &out) {
	if (in==eq_id) out = EQ;
	else if (in==lt_id) out = LT;
	else if (in==gt_id) out = GT;
	else if (in==ne_id) out = DF;
	else if (in==ss_id) out = IN;
	else if (in==pf_id) out = PF;
	else {
		cerr << "Unsupported operator type" << endl;
		exit(1);
	}
}

void Translator::translateFilter(simple_filter *in, CudaFilter *out) {
	simple_filter::iterator *it = in->first();
	out->numConstraints=0;
	do {
		if (out->numConstraints>MAX_CONSTR_NUM) {
			cerr << "Overcoming the maximum number of constraints" << endl;
			exit(1);
		}
		for (unsigned int i=0; i<it->name().length(); i++) {
			out->constraints[out->numConstraints].name[i] = it->name()[i];
		}
		translateOp(it->op(), out->constraints[out->numConstraints].op);
		if (it->type()==int_id) {
			out->constraints[out->numConstraints].value.type = INT;
			out->constraints[out->numConstraints].value.intVal = it->int_value();
		} else if (it->type()==string_id) {
			out->constraints[out->numConstraints].value.type = STRING;
			for (unsigned int i=0; i<it->string_value().length(); i++) {
				out->constraints[out->numConstraints].value.stringVal[i] = it->string_value()[i];
			}
		} else {
			cerr << "Unsupported value type" << endl;
			exit(1);
		}
		out->numConstraints++;
	} while(it->next());
	delete it;
}
