#include <iostream>
#include <string>

#include <siena/attributes.h>

using namespace std;
using namespace siena;

class text_message: public Message {
    // this class implements the message interface
    // ...
public:
    // this method reads message from an input stream
    istream & read_from(istream & i); 
    // this method returns a "text" for this message
    const string & get_text() const;
};

class simple_handler: public MatchHandler {
private:
    ostream * channels[];
    const text_message * msg;
    int counter;

public:
    simple_handler(ostream * oc[]): channels(oc) {}

    setup_message(const text_message * m, int c) {
	msg = m;
	counter = c;
    }
    virtual bool output(if_t i);
};

bool simple_handler::output(if_t i) {
    //
    // we output the message text to the output stream 
    // associated with the matching interface
    //
    *(os[i]) << msg->get_text() << endl;
    if (--c <= 0) { 
	return true;		// enough processing for this message:
    } else {			// true forces the matching process to end.
	return false;
    }
}

int main(int argc, char * argv[]) {
    text_message m;
    ostream * output_channels[10];

    AttributesFIB * FT = FwdTable.create();
    //
    // here we initialize FT with some rules
    // FT.ifconfig(...);
    // FT.ifconfig(...);
    // ...
    //
    // we also initialize our output channels
    // output_channel[0] = ...;
    // output_channel[1] = ...;
    // ...
    simple_handler h(output_channels);

    while(m.read_from(cin)) {	// then we start reading messages from stdin
	h.setup(&m, 5);		// and we process each message through FT
	FT->match(m, h);	// using our simple handler.
    }
    delete(FT);
    return 0;
}
