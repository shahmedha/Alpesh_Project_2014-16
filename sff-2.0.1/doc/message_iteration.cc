#include <iostream>
#include <siena/attributes.h>

using namespace std;
using namespace siena;

void print_message(const Message & m) {

    Message::Iterator * i = m.first();
    String s;

    if (i != NULL) {
	do {
	    cout << "attribute: " << i->name().to_string() << endl;
	    switch (i->type()) {
	    case STRING: cout << "string = " << i->string_value().to_string(s); break;
	    case INT: cout << "int = " << i->int_value(); break;
	    case BOOL: cout << "bool = " << i->bool_value(); break;
	    case DOUBLE: cout << "double = " << i->double_value(); break;
	    case ANYTYPE: 
		cout << "(generic value)" << endl; 
		cout << "Generic values are used in Constraints, not Messages." << endl; 
		break;
	    }
	    cout << endl;
	} while (i->next());
	delete(i);
    } else {
	cout << "empty message!" << endl;
    }
}

