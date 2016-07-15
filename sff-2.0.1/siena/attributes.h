// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2002-2003 University of Colorado
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
#ifndef SIENA_ATTRIBUTES_H_INCLUDED
#define SIENA_ATTRIBUTES_H_INCLUDED

#include <exception>
#include <string>

/** \file attributes.h 
 *
 *  This header file defines the basic types of the Siena data model
 *  defined on \em attributes.  Within the \em attributes model, data
 *  is described using named attributes whose values are taken from a
 *  fixed set of basic types, including numbers, strings, etc.  This
 *  file defines messages with attributes, and predicates with filters
 *  and constraints.
 **/

/** \namespace siena
 *
 *  @brief name space for Siena.
 *
 *  This namespace groups all the types and functionalities associated
 *  with the Siena system, including:
 *
 *  <ol>
 * 
 *  <li>The \em attributes data model, including its atomic types,
 *  attributes, constraints, messages, filters, and predicates.
 *
 *  <li>The \em tags data model, including tags, tag sets, and lists
 *  of tag sets.
 *
 *  <li>The forwarding modules, including all forwarding schemes and
 *  algorithms.
 *
 *
 *  <li>The routing modules.
 *  </ol>
 **/
namespace siena {
/** @brief operator identifier for the attributes-based data model.
 *
 *  Siena defines some of the most common operators (or binary
 *  relations) for the atomic types (strings, numbers, etc.) of the
 *  attributes data model. OperatorId is the type of operator
 *  identifiers.
 **/
enum OperatorId {
    /** @brief equals. 
     *
     *  Ordinary equality relation for basic types.
     **/ 
    EQ			= 1, 
    /** @brief less than.
     *
     *  Less-than ordering relation. Integers and doubles are ordered
     *  as usual, strings are ordered lexicographically. For booleans,
     *  \em false is less-than \em true.
     **/
    LT			= 2, 
    /** @brief greater than. 
     * 
     *  @see LT
     **/
    GT			= 3, 
    /** @brief suffix.
     *	
     *	suffix. <em>x</em> <code>SF</code> <em>y</em> is true when
     *	<em>y</em> is a suffix of <em>x</em>.  For example,
     *	<code>"software" SF "ware" == true</code>, while
     *	<code>"software" SF "w" == false</code>. <code>SF</code>
     *	is defined for strings only.
     **/
    SF			= 4,
    /** @brief prefix
     *
     *	prefix.  <em>x</em> <code>PF</code> <em>y</em> is true if
     *	<em>y</em> is a prefix of <em>x</em>.  For example,
     *	<code>"software" PF "soft" == true</code>, while
     *	<code>"software" PF "of" ==
     *	false</code>. <code>PF</code> is defined for strings only.
     **/
    PF			= 5,
    /** @brief contains substring
     *
     *	contains substring. <em>x</em> <code>SS</code> <em>y</em>
     *	is true if <em>y</em> is a substring of <em>x</em>.  For
     *	example, <code>"software" SS "war" == true</code>, while
     *	<code>"software" SS "hard" ==
     *	false</code>. <code>SS</code> is defined for strings only.
     **/
    SS			= 6,

    /** @brief any value
     *
     *  Tests only the existence of a given attribute.
     **/
    ANY			= 7,
    /** @brief not equal 
     * 
     *  Matches different values.  Notice, however, that a constraint
     *  with this operator would still require that the attribute
     *  exist, and that the values be of compatible types.
     **/
    NE			= 8,

    /** @brief matches regular expression
     *
     *	Matches a regular expression. <em>x</em> <code>RE</code> <em>y</em>
     *	is true if <em>y</em> is a regular expression that matches <em>x</em>.  
     *	For example, <code>"software" RE "([ot][fw])+a" == true</code>, while
     *	<code>"software" SS "o.{1,2}are$" == false</code>. <code>RE</code>
     *  is defined for strings only.  GNU extended regular expressions are
     *  completely supported except for backreferences -- that is, these are
     *  regular expression that can be described with a finite-state automaton
     *  or, equivalently, by a Chomsky type-3 grammars.
     **/
    RE			= 9
};

/** @brief type dentifiers in the attributes-based data model.
 *
 *  Siena defines some common basic data types, including numbers,
 *  booleans, and strings. TypeId defines identifiers for these types.
 *
 *  \sa Value::type()
 **/
enum TypeId {
    STRING		= 1,	/**< string identifier.	*/
    INT			= 2,	/**< integer identifier. */
    DOUBLE		= 3,	/**< double identifier.	*/
    BOOL		= 4,	/**< bool identifier.	*/
    ANYTYPE		= 5	/**< generic type identifier */
};

typedef long int	Int;	/**< integer type.	*/
typedef double		Double;	/**< double type.	*/
typedef bool		Bool;	/**< boolean type.	*/

/** @brief binary string type.
 *
 *  A reference to a contiguous sequence of bytes (8-bit characters).
 *  This is a very simple string representation.  The sequence is
 *  external to the String object and is referred to by a pair of
 *  pointers.  The first pointer (\link String::begin begin\endlink)
 *  points to the first character; the second pointer (\link
 *  String::end end\endlink) points to the first charcter past the end
 *  of the sequence.  A string may contain any character.  A string
 *  may also be empty when <code>begin == end</code>.
 *
 *  String objects are immutable for the purpose of the forwarding
 *  module, in the sense that the String class does not provide
 *  methods to modify the value of a String.
 **/
class String {
public:
    /** @brief pointer to the first character in the string. */
    const char * begin;
    /** @brief pointer to the first character past-the-end of the string. */
    const char * end;
    
    /** @brief constructs an empty string. */
    String() : begin(0), end(0) {}

    /** @brief constructs a string with the given begin and end pointers. */
    String(const char * b, const char * e) : begin(b), end(e) {}

    /** @brief constructs a string with the given char * interpreted
     *  as a C-style string.
     **/
    String(const char * s) 
	: begin(s), end(s) { while(*end != 0) ++end; }

    /** @brief constructs a string with the given pointer and length. */
    String(const char * s, int len) : begin(s), end(s + len) {}

    /** @brief constructs a copy of a given string. 
     *
     *  The copy is simply a copy of the reference.  In other words,
     *  this method does not allocate memory for the string value.
     *  Instead, the new string will point to the same sequence of the
     *  given string.
     **/
    String(const String & x) : begin(x.begin), end(x.end) {}

    /** @brief assigns this string to the given string. 
     *
     *  This method does not allocate memory.  Instead, this string
     *  will point to the same memory sequence pointed by the given
     *  string.
     **/
    String & operator = (const String x) { 
	begin= x.begin; end = x.end; return *this;
    }

    /** @brief assigns this string to the given sequence. 
     *
     *  This method does not allocate memory.  
     **/
    String & assign(const char * b, const char * e) { 
	begin = b; end = e; return *this;
    }

    /** @brief returns the size of the string. */
    unsigned int length() const { 
	return end - begin; 
    }

    /** @brief accessor method for strings. */
    const char &operator [] (int x) const { 
	return begin[x];
    }

    /** @brief equality test for strings. */
    bool operator == (const String & x) const;

    /** @brief lexicographical order relation. 
     *
     *  returns true iff this string lexicographically precedes the
     *  given string.
     **/
    bool operator < (const String & x) const;

    /** @brief lexicographical order relation. 
     *
     *  returns true iff this string lexicographically follows the
     *  given string.
     **/
    bool operator > (const String & x) const { 
	return x < *this;
    }

    /** @brief not equals
     **/
    bool operator != (const String & x) const { 
	return !(*this == x);
    }

    /** @brief prefix */
    bool has_prefix (const String & x) const;

    /** @brief suffix */
    bool has_suffix (const String & x) const;

    /** @brief substring
     **/
    bool has_substring (const String & x) const;

    /** @brief assigns this string value to a standard string.
     **/
    std::string & to_string(std::string & s) const;

    /** @brief returns this string value as a standard string value.
     **/
    std::string to_string() const;
};

/** @brief interface of a generic \em value in the attribute-based
 * data model.
 *
 *  This represents a typed value.  This is how one could print the value:
 * @code
 *  void print_siena_value(const siena::Value * v) {
 *      std::string s;
 *      switch(v->type()) {
 *      case siena::STRING: 
 * 	    cout << v->string_value().to_string(s) << endl;
 * 	    break;
 *      case siena::INT:
 * 	    cout << v->int_value() << endl;
 * 	    break;
 *      case siena::DOUBLE:
 * 	    cout << v->double_value() << endl;
 * 	    break;
 *      case siena::BOOL:
 * 	    cout << (v->bool_value() ? "true" : "false") << endl;
 * 	    break;
 *      case siena::ANYTYPE: 
 * 	    cout << "(generic value)" << endl;
 *          cout << "Generic values are used in Constraints, not Messages." << endl; 
 *      }
 *  }
 *  @endcode
 **/
class Value {
public:
    /** @brief virtual destructor */
    virtual ~Value() {};

    /** @brief returns the actual type of this value. 
     **/
    virtual TypeId type() const = 0;

    /** @brief returns this value as an integer. 
     *
     *  This method returns this value as an integer if the actual
     *  type (returned by type()) is INT.  The result is undefined if
     *  the actual type is not INT.
     *
     *  \sa TypeId, type().
     **/
    virtual Int int_value() const = 0;

    /** @brief returns this value as a string. 
     *
     *  This method returns this value as a string if the actual type
     *  (returned by type()) is STRING.  The result is undefined if
     *  the actual type is not STRING.
     *
     *  \sa TypeId, type().
     **/
    virtual String string_value() const = 0;

    /** @brief returns this value as a boolean. 
     *
     *  This method returns this value as a boolean if the actual type
     *  (returned by type()) is BOOL.  The result is undefined if the
     *  actual type is not BOOL.
     *
     *  \sa TypeId, type().
     **/
    virtual Bool bool_value() const = 0;

    /** @brief returns this value as a double. 
     *
     *  This method returns this value as a boolean if the actual type
     *  (returned by type()) is DOUBLE.  The result is undefined if
     *  the actual type is not DOUBLE.
     *
     *  \sa TypeId, type().
     **/
    virtual Double double_value() const = 0;
};

/** @brief interface of a generic <em>attribute</em> in the attribute-based data model.
 *
 *  An attribute is a named <em>value</em>.
 *  \sa Value
 **/
class Attribute : public Value {
public:
    /** @brief virtual destructor */
    virtual ~Attribute() {};
    /** @brief returns the name of this attribute. */
    virtual String name() const = 0;
};

/** @brief interface of a generic \em constraint in the attribute-based data model.
 *
 *  A constraint is defined by a name, an operator, and a value.  A
 *  constraint expresses the condition that must be matched by a
 *  message.  Specifically, a constraint defined by name \em N,
 *  operator \em Op, and value \em V is matched by a message \em M if
 *  \em M contains an attribute named \em N and value \em X and any of
 *  the following conditions applies:
 *
 *  <ol>
 * 
 *  <li>\em Op == ANY and \em V.type() == ANYTYPE
 *
 *  <li>\em Op == ANY and \em V.type() == \em X.type()
 *
 *  <li>value \em X and value \em V (in this order) are in the
 *  relation defined by operator \em Op.  In general, this happens in the following cases:
 *
 *  <ul> 
 *
 *  <li>same type, matching values: \em X.type() == \em V.type() and the
 *  two values satisfy the semantics of the \em Op condition as
 *  defined intuitively.  For example, if \em Op is LT and \em
 *  X.type() == \em V.type() == STRING, then the constraint is
 *  satisfied if \em X.string_value() is less than \em
 *  V.string_value(), which means that \em X.string_value() precedes
 *  \em V.string_value() in lexicographical order.
 *  
 *  <li>compatible types, matching values: \em X.type() == INT and \em
 *  V.type() == DOUBLE (or vice versa) and \em X.int_value() \em Op
 *  \em V.double_value().
 *
 *  </ul>
 *
 *  Notice that some operators are type-specific (e.g., SF, PF, SS are
 *  defined for strings only).
 *
 *  </ol>
 *
 *  \sa OperatorId, Attribute, Value
 **/
class Constraint : public Attribute {
public:
    /** @brief virtual destructor */
    virtual ~Constraint() {};
    /** @brief returns the operator defined for this constraint. */
    virtual OperatorId op() const = 0;

    /** @brief applies this constraint to an attribute.
     *  
     *  @return \em true iff this constraint matches (i.e., covers)
     *  the given attribute.
     **/
    virtual bool covers(const Attribute & a) const;
};

/** @brief error condition for an invalid constraint.
 **/
class BadConstraint : public std::exception {
    std::string err_descr;

public:
    /** @brief creates a bad_constraint exception with the given
     *  description
     **/
    BadConstraint(const std::string & e) 
	: err_descr(e) {}

    virtual ~BadConstraint() throw() {}

    /** @brief gives some details on the nature of the problem.
     **/
    virtual const char* what() const throw() {
	return err_descr.c_str();
    }
};

/** @brief interface of a generic \em message in the attribute-based
 *  data model.
 *
 *  A \em message is a set of attributes.  Individual attributes can
 *  be accessed sequentially through a Message::Iterator, or directly
 *  by name.
 **/
class Message {
public:
    /** @brief virtual destructor */
    virtual ~Message() {};
    /** \example message_iteration.cc 
     *  
     *  This example shows how to iterate through a message.  The same
     *  iteration model used in this example can be also be applied to
     *  Filter and Predicate objects.
     **/
    /** @brief interface of a generic message iterator.
     *
     *  provides sequential access to the attributes of a message.
     **/
    class Iterator : public Attribute {
    public:
	/** @brief moves this iterator to the next attribute in the sequence.
	 *
	 *  @return <code>true</code> if the element pointed to by
	 *  this iterator <em>before</em> this call to next() is not
	 *  the last attribute in its message.  In this case, after
	 *  this call to next(), this iterator will point to the next
	 *  attribute.
	 *  <br><code>false</code> if the element pointed to by this
	 *  iterator <em>before</em> this call to next() is either the
	 *  the last attribute in its message, or an invalid
	 *  attribute.  In this case, after this call to next(), this
	 *  iterator will point to an invalid attribute.
	 **/
	virtual bool next() = 0;
	/** @brief iterator destructor. */
	virtual ~Iterator() {};
    };

    /** @brief returns an iterator over this message. 
     *  
     *  The iterator returned by this method must define a complete
     *  iteration through the message.  The order of the iteration is
     *  implementation-dependent.
     *
     *  @return iterator pointing to the first attribute in this
     *  message, or \c NULL if this message is empty.
     **/
    virtual Iterator * first() const = 0;

    /** @brief provides direct access to a given attribute. 
     *
     *  Finds an attribute in a message based on the attribute's name.
     *  Applications must take care of deallocating the objects
     *  returned by this method.
     *
     *  @return the given attribute in this message, or \c NULL if
     *  such attribute does not exist.
     **/
    virtual Attribute * find(const String &) const = 0;

    /** @brief test existence of a given attribute. */
    virtual bool contains(const String &) const = 0;
};

/** @brief interface of a generic <em>filter</em> in the attribute-based data model.
 *
 *  A <em>filter</em> is a set of constraint representing a logical
 *  <em>conjunction</em> of elementary conditions.  Individual
 *  constraints can be accessed sequentially through a
 *  Filter::Iterator.
 **/
class Filter {
public:
    /** @brief virtual destructor */
    virtual ~Filter() {};
    /** @brief interface of a generic filter iterator.
     *
     *  provides sequential access to the constraints of a filter.
     **/
    class Iterator : public Constraint {
    public:
	/** @brief moves this iterator to the next attribute in the sequence.
	 *
	 *  @return <code>true</code> if the constraint pointed to by
	 *  this iterator <em>before</em> this call to next() is not
	 *  the last one in its filter.  In this case, after this
	 *  call to next(), this iterator will point to the next
	 *  constraint.
	 *  <br><code>false</code> if this iterator points to the last
	 *  constraint or to the end of the sequence.  In this case,
	 *  after this call to next(), this iterator will point to an
	 *  invalid constraint representing the end of the sequence.
	 **/
	virtual bool next() = 0;
	/** @brief destructor */
	virtual ~Iterator() {};
    };
    /** @brief returns an iterator over this filter. 
     *  
     *  @return iterator pointing to the first constraint in this
     *  filter, or \c NULL if this filter is empty.
     **/
    virtual Iterator * first() const = 0;

    /** @brief applies this filter to a message. 
     *  
     *  This is a naive implementation of the matching function.
     *
     *  @return true iff this filter matches (i.e., covers) the given
     *  message.
     **/
    virtual bool covers(const Message & m) const;
};

/** @brief interface of a generic <em>predicate</em> in the attribute-based data model.
 *
 *  A <em>predicate</em> is a set of filters representing a logical
 *  <em>disjunctions</em> of <em>conjunctions</em> of elementary
 *  conditions.  Individual conjunctions can be accessed sequentially
 *  through a predicate::iterator.
 **/
class Predicate {
public:
    /** @brief virtual destructor */
    virtual ~Predicate() {};

    /** @brief interface of a generic predicate iterator.
     *
     *  provides sequential access to the filters in a predicate.
     **/
    class Iterator : public Filter {
    public:
	/** @brief moves this iterator to the next filter in its predicate.
	 *
	 *  @return <em>true</em> if the filter pointed to by this
	 *  iterator <em>before</em> this call to next() is not the
	 *  last one in its predicate.  In this case, after this call
	 *  to next(), this iterator will point to the next
	 *  filter.
	 *  <br><code>false</code> if this iterator points to the last
	 *  filter or to the end of the sequence.  In this case, after
	 *  this call to next(), this iterator will point to an
	 *  invalid filter representing the end of the sequence.
	 **/
	virtual bool next() = 0;
	/** @brief destructor */
	virtual ~Iterator() {};
    };

    /** @brief returns an iterator over this predicate. 
     *  
     *  @return iterator pointing to the first filter in this
     *  predicate, or \c NULL if this predicate is empty.
     **/
    virtual Iterator * first() const = 0;

    /** @brief applies this predicate to a message.
     *  
     *  This is a naive implementation of the matching function.
     *
     *  @return <em>true</em> iff this predicate matches (i.e.,
     *  covers) the given message.
     **/
    virtual bool covers(const Message & m) const;
};

} // end namespace siena

#endif
