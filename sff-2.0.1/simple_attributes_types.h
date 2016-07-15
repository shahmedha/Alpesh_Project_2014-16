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
#ifndef SIMPLE_ATTRIBUTES_TYPES_H
#define SIMPLE_ATTRIBUTES_TYPES_H

#include <list>
#include <map>

#include <siena/attributes.h>

class simple_value: public siena::Value {
public:
    virtual siena::TypeId type() const;
    virtual siena::Int int_value() const;
    virtual siena::String string_value() const ;
    virtual siena::Bool bool_value() const;
    virtual siena::Double double_value() const;

    simple_value();
    simple_value(siena::Int x);
    simple_value(siena::Bool x);
    simple_value(siena::Double x);
    simple_value(const siena::String & x);

private:
    siena::TypeId t;
    union {
	siena::Int i;
	siena::Bool b;
	siena::Double d;
    };
    siena::String s;
};

typedef std::map<siena::String, const simple_value *> attribute_map;

class simple_attribute : public siena::Message::Iterator {
public:
    virtual siena::String name() const;
    virtual siena::TypeId type() const;
    virtual siena::Int int_value() const;
    virtual siena::String string_value() const;
    virtual siena::Bool	bool_value() const;
    virtual siena::Double double_value() const;
    virtual bool next();

    simple_attribute(attribute_map::const_iterator b, 
		     attribute_map::const_iterator e);
private:
    attribute_map::const_iterator i;
    attribute_map::const_iterator end;
};

class simple_message: public siena::Message {
public:
    virtual Iterator * first() const;
    virtual Iterator * find(const siena::String & name) const;
    virtual bool contains(const siena::String & name) const;
    bool add(const siena::String & name, const simple_value * a);

    simple_message();

    virtual ~simple_message();

private:
    attribute_map attrs;
};

class simple_op_value: public simple_value {
public:
    virtual siena::OperatorId op() const;

    simple_op_value(siena::OperatorId xo);
    simple_op_value(siena::OperatorId xo, siena::Int x);
    simple_op_value(siena::OperatorId xo, siena::Bool x);
    simple_op_value(siena::OperatorId xo, siena::Double x);
    simple_op_value(siena::OperatorId xo, const siena::String & x);

private:
    siena::OperatorId o;
};

typedef std::multimap<siena::String, const simple_op_value *> constraint_map;

class simple_constraint: public siena::Filter::Iterator {
public:
    virtual siena::String name() const;
    virtual siena::TypeId type() const;
    virtual siena::Int int_value() const;
    virtual siena::String string_value() const;
    virtual siena::Bool	bool_value() const;
    virtual siena::Double double_value() const;
    virtual siena::OperatorId op() const;
    virtual bool next();
    simple_constraint(constraint_map::const_iterator b, 
		      constraint_map::const_iterator e);
private:
    constraint_map::const_iterator i;
    constraint_map::const_iterator end;
};

class simple_filter: public siena::Filter {
public:
    virtual Iterator * first() const;
    void add(const siena::String name, 
	     const simple_op_value * v);

    simple_filter();
    virtual ~simple_filter();

private:
    constraint_map constraints;
};

typedef std::list<simple_filter *> filter_list;

class simple_predicate_i: public siena::Predicate::Iterator {
public:
    virtual Filter::Iterator * first() const;
    virtual bool next();

    simple_predicate_i(filter_list::const_iterator b,
		       filter_list::const_iterator e);
private:
    filter_list::const_iterator i;
    filter_list::const_iterator end;
};

class simple_predicate : public siena::Predicate {
public:
    virtual Iterator * first() const;
    void add(simple_filter * v);
    simple_filter * last_filter();

    simple_predicate();
    virtual ~simple_predicate();

private:
    filter_list filters;
};

#endif
