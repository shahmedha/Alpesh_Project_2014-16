// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2001-2003 University of Colorado
//  Copyright (C) 2005-2013 Antonio Carzaniga
//
//  Siena is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//  
//  Siena is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//  
//  You should have received a copy of the GNU General Public License
//  along with Siena.  If not, see <http://www.gnu.org/licenses/>.
//
/** \mainpage Siena Fast Forwarding API Documentation

This documentation describes the application programmer interface of
the <a href="http://www.inf.usi.ch/carzaniga/siena/">Siena</a> Fast
Forwarding module (SFF).  Technical documentation including the design
of the forwarding algorithm and the general architecture of the
forwarding module is available in a number of <a
href="http://www.inf.usi.ch/carzaniga/cbn/index.html#documents">technical
papers</a> and within the source code.

The Siena Fast Forwarding module implements a generic forwarding
engine for a content-based router.  The engine is intended to be
generic in the sense that it is intended to support multiple kinds of
forwarding schemes and algorithms, and multiple data/addressing
models.  By forwarding <em>scheme</em> here we mean the way the
forwarding information base (FIB) is structured as well as how the FIB
is used to forward messages.  An <em>addressing/data</em> model
instead defines the form and semantics of the data/addresses used to
set up the FIB, as well as the form and semantics of the
data/addresses carried by messages and used as the main input in the
forwarding function.

The current module implements one forwarding scheme with two
particular data/addressing models.  These two implementations are
abstracted by the two interface classes \link siena::AttributesFIB
AttributesFIB\endlink and \link siena::TagsFIB TagsFIB\endlink.  Other
forwarding schemes and data models can be implemented by extending the
\link siena::FIB FIB base class\endlink or even independently of that
interface.

The current available forwarding scheme is one in which router
interfaces are associated with predicates, and where an incoming
message is forwarded to all interfaces associated with matching
predicates except those indicated in a special exclusion set.  This
forwarding scheme can implement a simple forwarding on a tree, as well
as a more general forwarding scheme such as the Combined Broadcast-
and Content-Based routing (CBCB, see A. Carzaniga, M.J. Rutherford,
and A.L. Wolf, <a
href="http://www.inf.usi.ch/papers/crw_infocom04.pdf">A Routing Scheme
for Content-Based Networking</a>, INFOCOM 2004).

The current available data models are:

<ul>

<li>one based on <em>attributes</em>, where messages are described by
    a set of <em>attributes</em> and messages can be selected through
    combinations of constraints posed on the values of their
    describing attributes.  In particular, a forwarding table is a
    one-to-one association of <em>predicates</em> to
    <em>interfaces</em>.  A predicate is a <em>disjunction</em> of
    <em>conjunctions</em> of elementary <em>constraints</em>.  The
    forwarding table provides essentially one match function that
    takes a <em>message</em> <em>M</em> and outputs the set of
    interfaces associated with predicates matched by <em>M</em>.  This
    attribute-based data model is defined by \link siena::Value
    values\endlink, \link siena::Attribute attributes\endlink, \link
    siena::Message messages\endlink, \link siena::Constraint
    constraints\endlink, \link siena::Filter filters\endlink, and
    \link siena::Predicate predicates\endlink, which are presented in
    the \link attributes.h attributes.h header file\endlink.

<li>one based on <em>tags</em>, where messages are described by sets
    and messages can be selected when their tag sets include one or
    more of a given list of tag sets.  In particular, a forwarding
    table is a one-to-one association of <em>predicates</em> to
    <em>interfaces</em>, and a predicate is a list of sets of tags
    <em>S<sub>1</sub>, S<sub>2</sub>,...,S<sub>n</sub></em> such that
    a message <em>M</em> described by a set of tags
    <em>S<sub>M</sub></em> would match the predicate if
    <em>S<sub>M</sub></em> contains <em>S<sub>i</sub></em> for some
    <em>i</em>.  This data model is defined by \link siena::Tag
    tags\endlink, \link siena::TagSet tag sets\endlink, \link
    siena::TagSetList tag-set lists \endlink, which are presented in
    the \link tags.h tags.h header file\endlink.

</ul>

The API of the forwarding table is defined by the \link
siena::AttributesFIB AttributesFIB\endlink and \link siena::TagsFIB
TagsFIB\endlink interfaces.  Eight concrete implementations are
currently available for the attribute-based model (\link
siena::AttributesFIB AttributesFIB\endlink) each implementing a
different matching algorithm: \link siena::FwdTable FwdTable\endlink,
\link siena::BTable BTable\endlink, \link siena::BTrieTable
BTrieTable\endlink, \link siena::SortedBTable SortedBTable\endlink,
\link siena::BXTable BXTable\endlink, \link siena::BCTable
BCTable\endlink, \link siena::BDDBTable BDDBTable\endlink, and \link
siena::ZDDBTable ZDDBTable\endlink.  These last six implementations
(BTable, SortedBTable, BXTable, BCTable, BDDBTable, and ZDDBTable) are
based on encoded predicates and messages.  Because of the nature of
the encoding, these implementations are likely to generate false
positives.  Therefore, applications using these implementations should
be designed to compensate for this behavior. (See \link siena::BTable
BTable\endlink for more details.)  The library also provides two
concrete implementations of the \link siena::TagsFIB TagsFIB\endlink
interfaces with the \link siena::TagsTable TagsTable\endlink and \link
siena::TTable TTable\endlink classes.

The SFF module is designed to operate on application data, namely on
messages and predicates, regardless of the internal or external format
of those messages and predicates.  To make such application data
accessible, SFF defines a minimal interface through which messages and
predicates are read by the SFF algorithms.  The application programmer
must therefore implement this interface.

Specifically, the interface to application data in the
attributes-based data model consists of the \link siena::Value
Value\endlink, \link siena::Attribute Attribute\endlink, \link
siena::Message Message\endlink, \link siena::Constraint
Constraint\endlink, \link siena::Filter Filter\endlink, and \link
siena::Predicate Predicate\endlink interfaces and their iterators.
Similarly, the tags-based data model is defined by the \link
siena::Tag Tag\endlink, \link siena::TagSet TagSet\endlink, and \link
siena::TagSetList TagSetList\endlink interfaces and their iterators.
A complete sample implementation of these interfaces is available from
the <a href="examples.html">Example Section</a>.  Specifically, see
\link simple_attributes_types.h\endlink and \link
simple_attributes_types.cc\endlink for the attributes-based data
model, and \link simple_tags_types.h\endlink and \link
simple_tags_types.cc\endlink for the tags-based data model.

In addition to its application programming interface, the Siena Fast
Forwarding Module can be used through an external \em driver
program. The command-line options of the driver program are documented
\link drivercommandline here\endlink.  The driver program reads
commands from an input file or interactively thorugh the standard
input. The commands understood by the driver include essentially all
the operations of the forwarding table for both the attribute-based
and tag-based data model.  In particular, the driver can associate
predicates to interfaces, and can request the processing of messages.
Examples including the main commands accepted by the driver are
available for both the \link driver-attributes.input
attributes-based\endlink and \link driver-tags.input
tags-based\endlink data models from the <a
href="examples.html">Example Section</a>.  The syntax of the input to
the driver is also documented in a separate page on the \ref
driversyntax.

**/

/** \page drivercommandline Driver Program: Command-line Parameters

This page documents the command-line parameters of the driver program.
Notice that some option may not be available due to the particular
compilation configuration.

The driver program, called \c sff, has the following general command-line options:

\c sff \c -f \em algorithm [\c -p] [\c -i] [\c -i \em ranges] [\c -o \em stats-format] [\c -O \em filename] [\c --] [\em filename ...]

The driver program reads from the given file names, one after the
other, or from the standard input if no input file name is given.
Below is the detailed documentation of each command-line option.

<ul>

<li>\c -f \em algorithm :  selects a forwarding algorithm/table.
       The following algorithm identifiers are available:
       <ul>
       <li>\c d :  uses the \link siena::FwdTable FwdTable\endlink algorithm (default).
       <li>\c b :  uses the \link siena::BTable BTable\endlink algorithm.
       <li>\c t :  uses the \link siena::BTrieTable BTrieTable\endlink algorithm.
       <li>\c s :  uses the \link siena::SortedBTable SortedBTable\endlink algorithm.
       <li>\c X :  uses the \link siena::BXTable BXTable\endlink algorithm.
       <li>\c c :  uses the \link siena::BCTable BCTable\endlink algorithm.
       <li>\c v :  uses the \link siena::BVTable BVTable\endlink algorithm.
       <li>\c B :  uses the \link siena::BDDBTable BDDBTable\endlink algorithm.
       <li>\c Z :  uses the \link siena::ZDDBTable ZDDBTable\endlink algorithm.
       <li>\c T :  uses the \link siena::TagsTable TagsTable\endlink algorithm.
       <li>\c Tt :  uses the \link siena::TTable TTable\endlink algorithm.
       </ul>


<li>\c -q : quiet.  Suppresses matching output.
<li>\c -c : prints only the total number (count) of interfaces matched by each message.
<li>\c -p : prints statistics after processing each input file.
<li>\c -i \em ranges :  only enable the given interfaces (e.g., 1-3,7).
<li>\c -O \em filename : writes results and statistics on the given output file.
<li>\c -o <stats-format>  uses the given output format for statistics.
       The \em stats-format parameter is a printf-style format string.
       The following format directives are defined:
       <ul>
       <li>\c %i : interface count.
       <li>\c %n : message count.
       <li>\c %c : constraint count.
       <li>\c %f : filter count.
       <li>\c %a : attribute count.
       <li>\c %w : tag count.
       <li>\c %W : tagset count.
       <li>\c %m : number of messages matching at least one interface.
       <li>\c %M : total number of matches.
       <li>\c %s : size of the forwarding table in bytes.
       <li>\c %S : total memory allocated by the forwarding table in bytes.
       <li>\c %T\em x : timers in milliseconds, where \em x can be:
		<ul>
		<li>\c t : sff library functions
		<li>\c p : parsing
                <li>\c i : \link siena::AttributesFIB::ifconfig ifconfig\endlink
		<li>\c c : \link siena::FIB::consolidate consolidate\endlink
		<li>\c m : \link siena::AttributesFIB::match matching\endlink
		<li>\c e : message encoding
		<li>\c s : string index
		<li>\c f : \link siena::MatchHandler forwarding\endlink
		</ul>
       </ul>

       The default format for statistics is this:

       \code
       "i=%i f=%f c=%c n=%n a=%a w=%w W=%W m=%m M=%M s=%s\n"
       \endcode

       plus the following, in case the library is configured (at
       compile-time) with high-precision timers:

       \code
	    "timers (milliseconds):\n"
	    "\tsff =\t\t%Tt\n"
	    "\tparser =\t%Tp\n"
	    "\tifconfig =\t%Ti\n"
	    "\tconsolidate =\t%Tc\n"
	    "\tencoding =\t%Te\n"
	    "\tmatch =\t\t%Tm\n"
	    "\tstring match =\t%Ts\n"
	    "\tforward =\t%Tf\n"
       \endcode
</ul>

**/

/** \example driver-attributes.input

Example of input commands for the driver of the forwarding table.
The syntax of commands is somehow similar to the syntax of C,
especially that of literal values.  

**/

/** \example driver-tags.input

Example of input commands for the driver of the forwarding table.
The syntax of commands is somehow similar to the syntax of C,
especially that of literal values.  

**/

/** \page driversyntax Driver Program: Input Format and Commands

This page contains a (almost complete) specification of the syntax of
the driver program, as well as a documentation of its control
commands.  An example is also available from the <a
href="examples.html">Example Section</a>:

<dl>
<dt><em>DriverInput</em><dd>
    ( \em Statement <code>;</code> )*

    <dt><em>Statement</em><dd>
    \em IFConfig
    <br>\em Select
    <br>\em ControlCommand

    <dt id="IFConfig"><em>IFConfig</em><dd>
  \c ifconfig \em number \em Predicate
  <br>\c ifconfig \em number \em TagSetList

  <dt><em>Predicate</em><dd>
    \em Constraint
    <br>\em Predicate \em Or \em Constraint
    <br>\em Predicate \em And \em Constraint

<dt><em>Or</em><dd> <code>||</code> <br><code>|</code> <br><code>\\/</code>

<dt><em>And</em><dd> <code>&&</code> <br><code>,</code> <br><code>/\\</code>

<dt><em>Constraint</em><dd> 
   \em Name \em Operator \em LiteralValue
   <br>\em Name \c any \em Type &nbsp;// any-value constraint

<dt><em>Name</em><dd>
   {<code>A-Za-z</code>}{<code>A-Za-z/.-</code>}*

<dt><em>Operator</em><dd>
   <code>=</code><br><code>!=</code><br><code>&lt;</code> <br><code>&gt;</code>
   <br><code>=*</code> <em>&nbsp;// prefix</em>  
   <br><code>*=</code> <em>&nbsp;// suffix</em>
   <br><code>**</code> <em>&nbsp;// substring</em>

<dt><em>LiteralValue</em><dd>
   <em>C-style integer literal</em> 
   <br><em>C-style string literal</em> 
   <br><em>C-style double literal</em> 
   <br><code>true</code>
   <br><code>false</code>

<dt><em>Number</em><dd>
   <em>C-style integer literal</em> 

<dt><em>Type</em><dd> 
   <code>integer</code>
   <br><code>string</code>
   <br><code>boolean</code>
   <br><code>double</code>
   <br><code>any</code>

<dt><em>TagSetList</em><dd>
    <code>{</code> TagSet <code>}</code>
    <br>\em TagSetList [<code>,</code>] <code>{</code> \em TagSet <code>}</code>

<dt><em>TagSet</em><dd>
    \em Tag
    <br>\em TagSet [<code>,</code>] \em Tag

<dt><em>Tag</em><dd>
    \em Name
    <br><em>C-style string literal</em>

<dt><em>Select</em><dd>
  <code>select</code> \em Message <code>;</code>
  <br><code>select</code> <code>{</code> TagSet <code>}</code> <code>;</code>

<dt><em>Message</em><dd>
   \em Attribute
   <br>\em Message \em Attribute

<dt><em>Attribute</em><dd>
   \em Name <code>=</code> \em LiteralValue

<dt><em>ControlCommand</em><dd>
<ul>
    <li>\c statistics [\em format]
    <br>
    prints the current statistics using the given format string, or the
    preconfigured format string if none is given.

    <li>\c set \c statistics_only <code>=</code> \c true
    <br>
    only reads and counts the input predicates and messages, without
    compiling or a forwarding table.  This might be useful to count the
    number of filters, predicates, constraints, etc. in a particular
    file.

    <li>\c set \c preprocessing_limit <code>=</code> \em number
    <br>
    sets the number of preprocessing rounds for the FwdTable algorithm.

    <li>\c set \c algorithm <code>=</code> \em algorithm
    <br>
    uses the given algorithm for the forwarding table.
    known algorithms are:
    \c fwdtable,
    \c bddbtable,
    \c zddbtable,
    \c btable,
    \c btrietable,
    \c sorted_btable,
    \c bxtable,
    \c bctable,
    \c bvtable,
    \c ttable,
    \c tagstable,

    <li>\c timer \c start
    <br>
    starts the performance timers

    <li>\c timer \c stop
    <br>
    stops the performance timers

    <li>\c output \c on
    <br>
    activates the normal output of the matcher 

    <li>\c output \c off
    <br>
    suppresses the normal output of the matcher

    <li>\c output <code>&gt;</code> [\em filename]
    <br>
    redirects the output of the matcher to the given file, or to
    standard output if a name is not given

    <li>\c clear
    <br>
    clears the forwarding table.  This is necessary to reconstruct the
    table (from scratch).  This operation also releases the memory allocated
    for the forwwarding table.

    <li>\c clear \c recycle
    <br>
    clears the forwarding table without releasing the allocated memory.
    This means that the memory will be recycled for the new content of
    the table.

    <li>\c consolidate
    <br>
    consolidates the current value of the forwarding table.  This is
    *necessary* before the forwarding table can be used for matching.
</ul>
</dl>

**/

/** \example simple_attributes_types.h

This is a header file defining a complete implementation of the Siena
attributes-based data model.  In particular, this file shows how to
implement the siena::Value, siena::Attribute, siena::Message,
siena::Constraint, siena::Filter, and siena::Predicate interfaces.
The implementation of the classes declared in this file are
implemented in the \link simple_attributes_types.cc
simple_attributes_types.cc\endlink source file.

**/

/** \example simple_attributes_types.cc

This is a complete implementation of the Siena attributes-based data
model.  In particular, this file shows how to implement the
siena::Value, siena::Attribute, siena::Message, siena::Constraint,
siena::Filter, and siena::Predicate interfaces.  This implementation
(definitions) correspond to the declaration in the \link
simple_attributes_types.h simple_attributes_types.h\endlink header
file.

**/

/** \example simple_tags_types.h

This is a header file defining a complete implementation of the Siena
tags-based data model.  In particular, this file shows how to
implement the siena::TagSet, siena::TagSetList interfaces.  The
implementation of the classes declared in this file are implemented in
the \link simple_tags_types.cc simple_tags_types.cc\endlink source
file.

**/

/** \example simple_tags_types.cc

This is a complete implementation of the Siena tags-based data model.
In particular, this file shows how to implement the siena::TagSet,
siena::TagSetList interfaces.  This implementation (definitions)
correspond to the declaration in the \link simple_tags_types.h
simple_tags_types.h\endlink header file.

**/

/** \example forwarding_messages.cc

This example shows how to set up a \link siena::MatchHandler match
handler\endlink for use with a \link siena::AttributesFIB
forwarding table\endlink.  In this example, the forwarding table
processes messages of type <code>text_message</code> delegating the
output function to an object of type <code>simple_handler</code>.
<code>simple_handler</code> puts out each message to a set of output
streams associated with the matching interfaces.  The handler also
implements a cut-off mechanism that limits the number of output
streams that a message is copied to.  In this particular example, this
limit is set to 5.  Notice that the handler is initialized with a
reference to message that is of the actual message type.  This allows
the handler to use specific access methods for its output function.
In this example, the handler uses the <code>get_text()</code> method
to obtain the output text from the message object.

**/
