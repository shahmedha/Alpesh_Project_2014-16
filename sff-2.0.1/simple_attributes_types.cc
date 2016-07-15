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
#include <siena/attributes.h>
#include "simple_attributes_types.h"

siena::TypeId simple_value::type() const {
    return t; 
}

siena::Int simple_value::int_value() const {
    return i; 
}

siena::String simple_value::string_value() const {
    return s; 
}

siena::Bool simple_value::bool_value() const {
    return b; 
}

siena::Double simple_value::double_value() const {
    return d; 
}

simple_value::simple_value() 
    : t(siena::ANYTYPE) {}

simple_value::simple_value(siena::Int x) 
    : t(siena::INT), i(x) {}

simple_value::simple_value(siena::Bool x) 
    : t(siena::BOOL), b(x) {}

simple_value::simple_value(siena::Double x) 
    : t(siena::DOUBLE), d(x) {}

simple_value::simple_value(const siena::String & x) 
    : t(siena::STRING), s(x) {}

simple_attribute::simple_attribute(attribute_map::const_iterator b, 
				   attribute_map::const_iterator e)  
    : i(b), end(e) {}

siena::String simple_attribute::name() const {
    return (*i).first; 
}

siena::TypeId simple_attribute::type() const {
    return (*i).second->type(); 
}

siena::Int simple_attribute::int_value() const {
    return (*i).second->int_value(); 
}

siena::String simple_attribute::string_value() const {
    return (*i).second->string_value(); 
}

siena::Bool simple_attribute::bool_value() const {
    return (*i).second->bool_value(); 
}

siena::Double simple_attribute::double_value() const {
    return (*i).second->double_value(); 
}

bool simple_attribute::next() { 
    if (i != end) ++i;
    return i != end;
}

simple_message::simple_message() 
    : attrs() {}

siena::Message::Iterator * simple_message::first() const {
    if (attrs.begin() == attrs.end()) {
	return 0;
    } else {
	return new simple_attribute(attrs.begin(), attrs.end());
    }
}

siena::Message::Iterator * simple_message::find(const siena::String & name) const {
    attribute_map::const_iterator i = attrs.find(name);
    if (i == attrs.end()) {
	return 0;
    } else { 
	return new simple_attribute(i, attrs.end());
    }
}

bool simple_message::contains(const siena::String & name) const {
    return attrs.find(name) != attrs.end();
}

bool simple_message::add(const siena::String & name, 
			 const simple_value * a) {
    return attrs.insert(attribute_map::value_type(name, a)).second;
}

simple_op_value::simple_op_value(siena::OperatorId xo) 
    : simple_value(), o(xo) {}

simple_op_value::simple_op_value(siena::OperatorId xo, 
				 siena::Int x) 
    : simple_value(x), o(xo) {}

simple_op_value::simple_op_value(siena::OperatorId xo, 
				 siena::Bool x) 
    : simple_value(x), o(xo) {}

simple_op_value::simple_op_value(siena::OperatorId xo, 
				 siena::Double x) 
    : simple_value(x), o(xo) {}

simple_op_value::simple_op_value(siena::OperatorId xo, 
				 const siena::String & x) 
    : simple_value(x), o(xo) {}

siena::OperatorId simple_op_value::op() const {
    return o; 
}

siena::String simple_constraint::name() const {
    return (*i).first; 
}

siena::TypeId simple_constraint::type() const {
    return (*i).second->type(); 
}

siena::Int simple_constraint::int_value() const {
    return (*i).second->int_value(); 
}

siena::String simple_constraint::string_value() const {
    return (*i).second->string_value(); 
}

siena::Bool simple_constraint::bool_value() const {
    return (*i).second->bool_value(); 
}

siena::Double simple_constraint::double_value() const {
    return (*i).second->double_value(); 
}

siena::OperatorId simple_constraint::op() const {
    return (*i).second->op(); 
}

bool simple_constraint::next() {
    if (i != end) ++i;
    return i != end;
}

simple_constraint::simple_constraint(constraint_map::const_iterator b, 
				     constraint_map::const_iterator e)
    : i(b), end(e) {}

siena::Filter::Iterator * simple_filter::first() const {
    if (constraints.begin() == constraints.end()) {
	return 0;
    } else {
	return new simple_constraint(constraints.begin(), 
				     constraints.end());
    }
}

void simple_filter::add(const siena::String name, 
			const simple_op_value * v) {
    constraints.insert(constraint_map::value_type(name, v));
}

simple_filter::simple_filter() : constraints() {}

siena::Filter::Iterator * simple_predicate_i::first() const {
    return (*i)->first();
}

bool simple_predicate_i::next() { 
    if (i != end) ++i;
    return i != end;
}

simple_predicate_i::simple_predicate_i(filter_list::const_iterator b,
				       filter_list::const_iterator e)
    : i(b), end(e) {}

siena::Predicate::Iterator * simple_predicate::first() const {
    if (filters.begin() == filters.end()) {
	return 0;
    } else {
	return new simple_predicate_i(filters.begin(), filters.end());
    }
}

void simple_predicate::add(simple_filter * v) {
    filters.push_back(v);
}

simple_filter * simple_predicate::last_filter() {
    return filters.back();
}

simple_predicate::simple_predicate() : filters() {}

simple_message::~simple_message() {
    for (attribute_map::const_iterator i = attrs.begin(); i != attrs.end(); ++i)
	if (i->second) delete(i->second);
}

simple_filter::~simple_filter() {
    for (constraint_map::const_iterator i = constraints.begin(); 
	 i != constraints.end(); ++i)
	if (i->second) delete(i->second);
}

simple_predicate::~simple_predicate() {
    for (filter_list::const_iterator i = filters.begin(); i != filters.end(); ++i)
	if (*i) delete(*i);
}

