// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2001-2003 University of Colorado
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
// $Id: simple_fwd_types.h,v 1.8 2010-03-11 15:16:41 carzanig Exp $
//
#ifndef SIMPLE_FWD_TYPES_H
#define SIMPLE_FWD_TYPES_H

#include <list>
#include <map>

#include "siena/types.h"

class simple_value: public siena::value {
public:
	virtual siena::type_id	type() const throw();
	virtual siena::int_t	int_value() const throw();
	virtual siena::string_t	string_value() const throw();
	virtual siena::bool_t	bool_value() const throw();
	virtual siena::double_t	double_value() const throw();

	simple_value() throw();
	simple_value(siena::int_t x) throw();
	simple_value(siena::bool_t x) throw();
	simple_value(siena::double_t x) throw();
	simple_value(const siena::string_t & x) throw();

private:
	siena::type_id t;
	union {
		siena::int_t i;
		siena::bool_t b;
		siena::double_t d;
	};
	siena::string_t s;
};

typedef std::map<siena::string_t, const simple_value *> attribute_map;

class simple_attribute : public siena::message::iterator {
public:
	virtual siena::string_t	name() const throw();
	virtual siena::type_id	type() const throw();
	virtual siena::int_t	int_value() const throw();
	virtual siena::string_t	string_value() const throw();
	virtual siena::bool_t	bool_value() const throw();
	virtual siena::double_t	double_value() const throw();
	virtual bool		next() throw();

	simple_attribute(attribute_map::const_iterator b,
			attribute_map::const_iterator e) throw();
private:
	attribute_map::const_iterator i;
	attribute_map::const_iterator end;
};

class simple_message: public siena::message {
public:
	virtual iterator *	first() const throw();
	virtual iterator *	find(const siena::string_t & name) const throw();
	virtual bool	contains(const siena::string_t & name) const throw();
	bool		add(const siena::string_t & name,
			const simple_value * a) throw();

	simple_message() throw();

	virtual ~simple_message() throw();

private:
	attribute_map attrs;
};

class simple_op_value: public simple_value {
public:
	virtual siena::operator_id	op() const throw();

	simple_op_value(siena::operator_id xo) throw();
	simple_op_value(siena::operator_id xo, siena::int_t x) throw();
	simple_op_value(siena::operator_id xo, siena::bool_t x) throw();
	simple_op_value(siena::operator_id xo, siena::double_t x) throw();
	simple_op_value(siena::operator_id xo, const siena::string_t & x) throw();

private:
	siena::operator_id o;
};

typedef std::multimap<siena::string_t, const simple_op_value *> constraint_map;

class simple_constraint: public siena::filter::iterator {
public:
	virtual siena::string_t	name() const throw();
	virtual siena::type_id	type() const throw();
	virtual siena::int_t	int_value() const throw();
	virtual siena::string_t	string_value() const throw();
	virtual siena::bool_t	bool_value() const throw();
	virtual siena::double_t	double_value() const throw();
	virtual siena::operator_id	op() const throw();
	virtual bool		next() throw();
	simple_constraint(constraint_map::const_iterator b,
			constraint_map::const_iterator e) throw();
private:
	constraint_map::const_iterator i;
	constraint_map::const_iterator end;
};

class simple_filter: public siena::filter {
public:
	virtual iterator *		first() const throw();
	void			add(const siena::string_t name,
			const simple_op_value * v) throw();

	simple_filter() throw();
	virtual ~simple_filter() throw();

private:
	constraint_map constraints;
};

typedef std::list<simple_filter *> filter_list;

class simple_predicate_i: public siena::predicate::iterator {
public:
	virtual filter::iterator * first() const throw();
	virtual bool		next() throw();

	simple_predicate_i(filter_list::const_iterator b,
			filter_list::const_iterator e) throw();
private:
	filter_list::const_iterator i;
	filter_list::const_iterator end;
};

class simple_predicate : public siena::predicate {
public:
	virtual iterator *		first() const throw();
	void			add(simple_filter * v) throw();
	simple_filter *		back() throw();

	simple_predicate() throw();
	virtual ~simple_predicate() throw();

private:
	filter_list filters;
};

#include "simple_fwd_types.icc"

#endif
