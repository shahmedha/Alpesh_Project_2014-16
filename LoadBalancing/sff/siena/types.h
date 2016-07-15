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
//  This program is free software; you can redistribute it and/or
//  modify it under the terms of the GNU General Public License
//  as published by the Free Software Foundation; either version 2
//  of the License, or (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307,
//  USA, or send email to one of the authors.
//
//
// $Id: types.h,v 1.16 2010-03-11 15:16:45 carzanig Exp $
//
#ifndef SIENA_TYPES_H
#define SIENA_TYPES_H

/** \file types.h 
 *
 *  This header file defines the basic types of the Siena data model.
 *  In essence, this file defines messages with their attributes, and
 *  predicates with their filters and constraints.
 **/

/** \namespace siena
 *
 *  @brief name space for Siena.
 *
 *  This namespace groups all the types and functionalities assiciated
 *  with Siena, including:
 *
 *  <ol>
 *  <li>the basic Siena model that defines atomic types, attributes,
 *  constraints, messages, filters, and predicates
 *
 *  <li>the forwarding module
 *
 *  <li>the routing module
 *  </ol>
 **/
namespace siena {
/** @brief operator identifier for <em>Siena</em>.
 *
 *  Siena defines some of the most common operators (or binary
 *  relations). <code>operator_id</code> is the type of operator
 *  identifiers.  The values for <code>operator_id</code> are those
 *  operators supported by the Siena Fast Forwarding algorithm.
 **/
enum operator_id {
	/** @brief equals */
	eq_id		= 1,
	/** @brief less than.
	 *
	 *  less than. Integers and doubles are ordered as usual, strings
	 *  are sorted in lexicographical order. For booleans,
	 *  <code>false</code> < <code>true</code>.
	 **/
	lt_id		= 2,
	/** @brief greater than */
	gt_id		= 3,
	/** @brief suffix
	 *
	 *	suffix. <em>x</em> <code>sf_id</code> <em>y</em> is true if
	 *	<em>y</em> is a suffix of <em>x</em>.  For example,
	 *	<code>"software" sf_id "ware" == true</code>, while
	 *	<code>"software" sf_id "w" == false</code>. <code>sf_id</code>
	 *	is defined for strings only.
	 **/
	sf_id		= 4,
	/** @brief prefix
	 *
	 *	prefix.  <em>x</em> <code>pf_id</code> <em>y</em> is true if
	 *	<em>y</em> is a prefix of <em>x</em>.  For example,
	 *	<code>"software" pf_id "soft" == true</code>, while
	 *	<code>"software" pf_id "of" ==
	 *	false</code>. <code>pf_id</code> is defined for strings only.
	 **/
	pf_id		= 5,
	/** @brief contains substring
	 *
	 *	contains substring. <em>x</em> <code>ss_id</code> <em>y</em>
	 *	is true if <em>y</em> is a substring of <em>x</em>.  For
	 *	example, <code>"software" ss_id "war" == true</code>, while
	 *	<code>"software" ss_id "hard" ==
	 *	false</code>. <code>ss_id</code> is defined for strings only.
	 **/
	ss_id		= 6,

	/** @brief any value
	 *
	 *  Tests only the existence of a given attribute.
	 **/
	any_id		= 7,
	/** @brief not equal */
	ne_id		= 8,

	/** @brief matches regular expression
	 *
	 *	matches regular expression. <em>x</em> <code>re_id</code> <em>y</em>
	 *	is true if <em>y</em> is a regular expression that matches <em>x</em>.
	 *	For example, <code>"software" re_id "([ot][fw])+a" == true</code>, while
	 *	<code>"software" ss_id "o.{1,2}are$" == false</code>. <code>re_id</code>
	 *  is defined for strings only.  GNU extended regular expressions are
	 *  completely supported except for backreferences -- that is, these are
	 *  regular expression that can be described with a finite-state automaton
	 *  or, equivalently, by a Chomsky type-3 grammars.
	 **/
	re_id		= 9
};

/** @brief type dentifiers in the <em>Siena</em> data model.
 *
 *  Siena defines some common basic data types, including numbers,
 *  booleans, and strings. <code>type_id</code> defines identifiers
 *  for these types.
 **/
enum type_id {
	string_id		= 1,	/**< string identifier.	*/
	int_id		= 2,	/**< integer identifier. */
	double_id		= 3,	/**< double identifier.	*/
	bool_id		= 4,	/**< bool identifier.	*/
	anytype_id		= 5	/**< generic type identifier */
};

typedef long long int	int_t;		/**< integer type.	*/
typedef double		double_t;	/**< double type.	*/
typedef bool		bool_t;		/**< boolean type.	*/

/** @brief string type.
 *
 *  a string is contiguous sequence of 8-bit characters.  The sequence
 *  is defined by a pair of pointers.  The first pointer
 *  <code>begin</code> points to the first character; the second
 *  pointer <code>end</code> points to the first charcter past the end
 *  of the sequence.  A string may contain any character.  A string
 *  may also be empty (when <code>begin == end</code>.)
 *
 *  <p><code>string_t</code>s are immutable strings, in the sense
 *  that <code>string_t</code> does not provide methods to modify the
 *  value of a <code>string_t</code>.  **/
class string_t {
public:
	/** @brief iterator type for <code>string_t</code>.
	 *
	 *  an iterator behaves like a pointer, and as such it can be
	 *  dereferenced, incremented, compared, etc.  However, iterators
	 *  are also specialized for strings and provide a few utility
	 *  methods, for example to retrieve the distance between an
	 *  iterator and the start of its corresponding <code>string_t</code>,
	 *  or to determine whether the end of the string has been reached.
	 *
	 *  <p><code>string_t</code>s are immutable strings, so this
	 *  type implements a <code>const</code> iterator.  **/
	class iterator {
		const string_t *s;
		const char *p;
		iterator (const string_t *str, const char *where) throw () :
			s (str), p (where) { }

	public:
		/** @brief constructs an iterator pointing to the beginning of
		 *  the given string.
		 **/
		static iterator begin (const string_t &s) throw () {
			return iterator (&s, s.begin);
		}

		/** @brief constructs an iterator pointing past the end of
		 *  the given string.
		 **/
		static iterator end (const string_t &s) throw () {
			return iterator (&s, s.end);
		}

		/** @brief constructs an iterator which is the copy of the given
		 *  one.
		 **/
		iterator (const iterator &i) throw () : s (i.s), p (i.p) { }

		/** @brief comparison operator for string iterators
		 *
		 *  returns true if the LHS iterator points earlier than RHS in
		 *  the same string; the return value is undefined if the iterators
		 *  do not belong to the same string.
		 **/
		bool operator <(const iterator &i) const throw () {
			return p < i.p;
		}

		/** @brief comparison operator for string iterators
		 *
		 *  returns true if the LHS iterator points no later than RHS in
		 *  the same string; the return value is undefined if the iterators
		 *  do not belong to the same string.
		 **/
		bool operator <=(const iterator &i) const throw () {
			return p <= i.p;
		}

		/** @brief comparison operator for string iterators
		 *
		 *  returns true if the LHS iterator points later than RHS in
		 *  the same string; the return value is undefined if the iterators
		 *  do not belong to the same string.
		 **/
		bool operator >(const iterator &i) const throw () {
			return p > i.p;
		}

		/** @brief comparison operator for string iterators
		 *
		 *  returns true if the LHS iterator points earlier than RHS in
		 *  the same string; the return value is undefined if the iterators
		 *  do not belong to the same string.
		 **/
		bool operator >=(const iterator &i) const throw () {
			return p >= i.p;
		}

		/** @brief equality operator for string iterators
		 *
		 *  returns true iff the LHS iterator points at the same place
		 *  as the RHS.
		 **/
		bool operator ==(const iterator &i) const throw () {
			return p == i.p;
		}

		/** @brief inequality operator for string iterators
		 *
		 *  returns false iff the LHS iterator points at the same place
		 *  as the RHS.
		 **/
		bool operator !=(const iterator &i) const throw () {
			return p != i.p;
		}

		/** @brief assignment operator for string iterators */
		iterator &operator =(const iterator &i) throw () {
			s = i.s;
			p = i.p;
			return *this;
		}

		/** @brief dereferencing operator for string iterators
		 *
		 *  returns a const lvalue for the character pointed to by the
		 *  iterator.
		 **/
		const char &operator *() const throw () {
			return *p;
		}

		/** @brief dereferencing operator for string iterators
		 *
		 *  returns a const lvalue for the character pointed to by the
		 *  iterator <code>*this + n</code>.
		 **/
		const char &operator [](int n) const throw () {
			return *(p + n);
		}

		/** @brief in-place addition operator for string iterators */
		iterator &operator += (int n) throw () {
			p += n; return *this;
		}

		/** @brief addition operator for string iterators */
		iterator operator +(int n) const throw () {
			return iterator (s, p + n);
		}

		/** @brief preincrement operator for string iterators */
		iterator &operator ++() throw () {
			++p; return *this;
		}

		/** @brief postincrement operator for string iterators */
		iterator operator ++(int) throw () {
			iterator i = *this; ++p; return i;
		}

		/** @brief in-place subtraction operator for string iterators */
		iterator& operator -= (int n) throw () {
			p -= n; return *this;
		}

		/** @brief subtraction operator for string iterators */
		iterator operator -(int n) const throw () {
			return iterator (s, p - n);
		}

		/** @brief difference operator for string iterators
		 *
		 *  returns <code>this->index () - i.index ()</code> if the two
		 *  iterators belong to the same string; the result is otherwise
		 *  undefined.
		 **/
		int operator -(const iterator &i) const throw () {
			return i.p - p;
		}

		/** @brief predecrement operator for string iterators */
		iterator &operator --() throw () {
			--p; return *this;
		}

		/** @brief postdecrement operator for string iterators */
		iterator operator --(int) throw () {
			iterator i = *this; --p; return i;
		}

		/** @brief return whether the iterator points to the end of the
		 *  string.
		 **/
		int at_end () const throw () {
			return p == s->end;
		}

		/** @brief get the relative position of the iterator in the string.
		 *
		 *  returns how many characters the iterator is far from
		 *  the beginning of the string.
		 **/
		int index () const throw () {
			return p - s->begin;
		}

		/** @brief set the relative position of the iterator in the string.
		 *
		 *  moves the iterator <code>index</code> characters past the
		 *  beginning of the string.
		 **/
		void set_index (int index) throw () {
			p = s->begin + index;
		}
	};

	/** @brief pointer to the first character in the string. */
	const char * begin;
	/** @brief pointer to the first character past-the-end of the string. */
	const char * end;

	/** @brief constructs an empty string. */
	string_t() throw() : begin(0), end(0) {}
	/** @brief constructs a string with the given begin and end pointers. */
	string_t(const char * b, const char * e) throw() : begin(b), end(e) {}
	/** @brief constructs a string with the given char * interpreted
	 *  as a C-style string.
	 **/
	string_t(const char * s) throw()
			: begin(s), end(s) { while(*end != 0) ++end; }
	/** @brief constructs a string with the given pointer and length. */
	string_t(const char * s, int len) throw() : begin(s), end(s + len) {}
	/** @brief constructs a copy of a given string.
	 *
	 *  The copy is simply a copy of the reference.  In other words,
	 *  this method does not allocate memory for the string value.
	 *  Instead, the new string will point to the same sequence of the
	 *  given string.
	 **/
	string_t(const string_t & x) throw() : begin(x.begin), end(x.end) {}
	/** @brief assigns this string to the given string.
	 *
	 *  This method does not allocate memory.  Instead, this string
	 *  will point to the same memory sequence pointed by the given
	 *  string.
	 **/
	string_t & operator = (const string_t x) throw() {
		begin= x.begin; end = x.end; return *this;
	}
	/** @brief assigns this string to the given sequence.
	 *
	 *  This method does not allocate memory.
	 **/
	string_t & assign(const char * b, const char * e) throw() {
		begin = b; end = e; return *this;
	}

	/** @brief returns the size of the string. */
	unsigned int length() const throw() { return end - begin; }

	/** @brief accessor method for strings. */
	const char &operator [] (int x) const throw() {
		return begin[x];
	}

	/** @brief equality test for strings. */
	bool operator == (const string_t & x) const throw() {
		if (length() != x.length()) return false;
		const char * i = begin;
		const char * j = x.begin;
		while(i != end)
			if (*i++ != *j++) return false;
		return true;
	}

	/** @brief lexicographical order relation.
	 *
	 *  returns true iff this string lexicographically precedes the
	 *  given string.
	 **/
	bool operator < (const string_t & x) const throw() {
		const char * i = begin;
		const char * j = x.begin;
		while(i != end) {
			if (j == x.end) return false;
			if (*i < *j) return true;
			if (*i > *j) return false;
			++i;
			++j;
		}
		return j != x.end;
	}

	/** @brief lexicographical order relation.
	 *
	 *  returns true iff this string lexicographically follows the
	 *  given string.
	 **/
	bool operator > (const string_t & x) const throw() {
		return x < *this;
	}

	/** @brief not equals
	 **/
	bool operator != (const string_t & x) const throw() {
		return !(*this == x);
	}

	/** @brief prefix
	 **/
	bool has_prefix (const string_t & x) const throw();

	/** @brief suffix
	 **/
	bool has_suffix (const string_t & x) const throw();

	/** @brief substring
	 **/
	bool has_substring (const string_t & x) const throw();
};

/** @brief interface of a generic <em>value</em> in the Siena data model.
 *
 *  A <em>value</em> is a typed value.
 **/
class value {
public:
	/** @brief virtual destructor */
	virtual			~value()			{};
	/** @brief returns the actual type identifier of this value.
	 *
	 *  \sa see type_id.
	 **/
	virtual type_id		type()				const = 0;
	/** @brief returns this value as an integer.
	 *
	 *  This method returns this value as an integer if the actual
	 *  type (returned by type()) is int_id.  The result is
	 *  undefined if the actual type is not int_id.
	 **/
	virtual int_t		int_value()			const = 0;
	/** @brief returns this value as a string.
	 *
	 *  This method returns this value as a string if the actual
	 *  type (returned by type()) is type_id::string_t.  The result is
	 *  undefined if the actual type is not string_t.
	 **/
	virtual string_t		string_value()			const = 0;

	/** @brief returns this value as a boolean.
	 *
	 *  This method returns this value as a boolean if the actual
	 *  type (returned by type()) is bool_id.  The result is
	 *  undefined if the actual type is not bool_id.
	 **/
	virtual bool_t		bool_value()			const = 0;
	/** @brief returns this value as a double.
	 *
	 *  This method returns this value as a boolean if the actual
	 *  type (returned by type()) is double_id.  The result is
	 *  undefined if the actual type is not double_id.
	 **/
	virtual double_t		double_value()			const = 0;
};

/** @brief interface of a generic <em>attribute</em> in the Siena data model.
 *
 *  An attribute is a named <em>value</em>.
 *  \sa see value
 **/
class attribute : public value {
public:
	/** @brief virtual destructor */
	virtual			~attribute()			{};
	/** @brief returns the name of this attribute. */
	virtual string_t		name()				const = 0;
};

/** @brief interface of a generic <em>constraint</em> in the Siena data model.
 *
 *  A <em>constraint</em> is defined by a name, an operator and a value.
 *
 *  \sa see operator_id, attribute, value
 **/
class constraint : public attribute {
public:
	/** @brief virtual destructor */
	virtual			~constraint()			{};
	/** @brief returns the operator defined for this constraint. */
	virtual operator_id		op()				const = 0;

	/** @brief applies this constraint to an attribute.
	 *
	 *  @return true iff this constraint matches (i.e., covers) the
	 *  given attribute.
	 **/
	virtual bool covers(const attribute & a) const throw();
};

/** @brief interface of a generic <em>message</em> in the Siena data model.
 *
 *  A <em>message</em> is a set of attributes.  Individual attributes
 *  can be accessed sequentially through a message::iterator, or
 *  directly by their name.
 **/
class message {
public:
	/** @brief virtual destructor */
	virtual			~message()			{};
	/** \example message_iteration.cc
	 *
	 *  This example shows how to iterate through a message.  The same
	 *  iteration model used in this example can be also be applied to
	 *  filter and predicate objects.
	 **/
	/** @brief interface of a generic message iterator.
	 *
	 *  provides sequential access to the attributes of a message.
	 **/
	class iterator : public attribute {
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
		virtual bool		next()				= 0;
		/** @brief iterator destructor. */
		virtual			~iterator()			{};
	};

	/** @brief returns an iterator over this message.
	 *
	 *  The iterator returned by this method must define a complete
	 *  iteration through the message.  The order of the iteration is
	 *  implementation-dependent.
	 *
	 *  @return iterator pointing to the first attribute in this
	 *  message, or <code>NULL</code> if this message is empty.
	 **/
	virtual iterator *		first()				const = 0;

	/** @brief provides direct access to a given attribute.
	 *
	 *  Finds an attribute in a message based on the attribute's name.
	 *  Applications must take care of deallocating the objects
	 *  returned by this method.
	 *
	 *  @return the given attribute in this message, or
	 *  <code>NULL</code> if such attribute does not exist.
	 **/
	virtual attribute *		find(const string_t &)		const = 0;

	/** @brief test existence of a given attribute. */
	virtual bool		contains(const string_t &)	const = 0;
};

/** @brief interface of a generic <em>filter</em> in the Siena data model.
 *
 *  A <em>filter</em> is a set of constraint representing a logical
 *  <em>conjunction</em> of elementary conditions.  Individual
 *  constraints can be accessed sequentially through a
 *  filter::iterator.
 **/
class filter {
public:
	/** @brief virtual destructor */
	virtual			~filter()			{};
	/** @brief interface of a generic filter iterator.
	 *
	 *  provides sequential access to the constraints of a filter.
	 **/
	class iterator : public constraint {
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
		virtual bool		next()				= 0;
		/** @brief destructor */
		virtual			~iterator()			{};
	};
	/** @brief returns an iterator over this filter.
	 *
	 *  @return iterator pointing to the first constraint in this
	 *  filter, or <code>NULL</code> if this filter is empty.
	 **/
	virtual iterator *		first() const = 0;

	/** @brief applies this filter to a message.
	 *
	 *  This is a naive implementation of the matching function.
	 *
	 *  @return true iff this filter matches (i.e., covers) the given
	 *  message.
	 **/
	virtual bool covers(const message & m) const throw();
};

/** @brief interface of a generic <em>predicate</em> in the Siena data model.
 *
 *  A <em>predicate</em> is a set of filters representing a logical
 *  <em>disjunctions</em> of <em>conjunctions</em> of elementary
 *  conditions.  Individual conjunctions can be accessed sequentially
 *  through a predicate::iterator.
 **/
class predicate {
public:
	/** @brief virtual destructor */
	virtual			~predicate()			{};

	/** @brief interface of a generic predicate iterator.
	 *
	 *  provides sequential access to the filters in a predicate.
	 **/
	class iterator : public filter {
	public:
		/** @brief moves this iterator to the next filter in its predicate.
		 *
		 *  @return <code>true</code> if the filter pointed to by this
		 *  iterator <em>before</em> this call to next() is not the
		 *  last one in its predicate.  In this case, after this call
		 *  to next(), this iterator will point to the next
		 *  filter.
		 *  <br><code>false</code> if this iterator points to the last
		 *  filter or to the end of the sequence.  In this case, after
		 *  this call to next(), this iterator will point to an
		 *  invalid filter representing the end of the sequence.
		 **/
		virtual bool		next()				= 0;
		/** @brief destructor */
		virtual			~iterator()			{};
	};

	/** @brief returns an iterator over this predicate.
	 *
	 *  @return iterator pointing to the first filter in this
	 *  predicate, or <code>NULL</code> if this predicate is empty.
	 **/
	virtual iterator *		first() const = 0;

	/** @brief applies this predicate to a message.
	 *
	 *  This is a naive implementation of the matching function.
	 *
	 *  @return true iff this predicate matches (i.e., covers) the
	 *  given message.
	 **/
	virtual bool covers(const message & m) const throw();
};

} // end namespace siena

#endif
