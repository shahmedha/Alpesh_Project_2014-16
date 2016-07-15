// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2004 University of Colorado
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

namespace siena {

bool Constraint::covers(const Attribute & a) const {
    switch(op()) {
    case EQ: 
	switch(type()) {
	case STRING: return (a.type() == STRING && a.string_value() == string_value());
	case INT: return (a.type() == INT && a.int_value() == int_value());
	case DOUBLE: return (a.type() == DOUBLE && a.double_value() == double_value());
	case BOOL: return (a.type() == BOOL && a.bool_value() == bool_value());
	case ANYTYPE: return true;
	}
    case LT:
	switch(type()) {
	case STRING: return (a.type() == STRING && a.string_value() < string_value());
	case INT: return (a.type() == INT && a.int_value() < int_value());
	case DOUBLE: return (a.type() == DOUBLE && a.double_value() < double_value());
	case BOOL: return (a.type() == BOOL && a.bool_value() < bool_value());
	case ANYTYPE: return true;
	}
    case GT:
	switch(type()) {
	case STRING: return (a.type() == STRING && a.string_value() > string_value());
	case INT: return (a.type() == INT && a.int_value() > int_value());
	case DOUBLE: return (a.type() == DOUBLE && a.double_value() > double_value());
	case BOOL: return (a.type() == BOOL && a.bool_value() > bool_value());
	case ANYTYPE: return true;
	}
    case SF: 
	switch(type()) {
	case STRING: return (a.type() == STRING && a.string_value().has_suffix(string_value()));
	case ANYTYPE: return true;
	case INT:
	case DOUBLE:
	case BOOL:
	    return false;
	}
    case PF: 
	switch(type()) {
	case STRING: return (a.type() == STRING && a.string_value().has_prefix(string_value()));
	case ANYTYPE: return true;
	case INT:
	case DOUBLE:
	case BOOL:
	    return false;
	}
    case SS: 
	switch(type()) {
	case STRING: return (a.type() == STRING && a.string_value().has_substring(string_value()));
	case ANYTYPE: return true;
	case INT:
	case DOUBLE:
	case BOOL:
	    return false;
	}
    case RE: 
	switch(type()) {
	case STRING: return false;
	case ANYTYPE: return true;
	case INT:
	case DOUBLE:
	case BOOL:
	    return false;
	}
    case ANY: 
	switch(type()) {
	case STRING: return (a.type() == STRING);
	case INT: return (a.type() == INT);
	case DOUBLE: return (a.type() == DOUBLE);
	case BOOL: return (a.type() == BOOL);
	case ANYTYPE: return true;
	}
    case NE: 
	switch(type()) {
	case STRING: return (a.type() == STRING && a.string_value() != string_value());
	case INT: return (a.type() == INT && a.int_value() != int_value());
	case DOUBLE: return (a.type() == DOUBLE && a.double_value() != double_value());
	case BOOL: return (a.type() == BOOL && a.bool_value() != bool_value());
	case ANYTYPE: return true;
	}
    }
    return false;
}

bool Filter::covers(const Message & m) const {
    Filter::Iterator * c = first();
    if (!c) return false;
    do {
	Attribute * a = m.find(c->name());
	if (!a) {
	    delete(c);
	    return false;
	} else if (! c->covers(*a)) {
	    delete(c);
	    delete(a);
	    return false;
	}
	delete(a);
    } while (c->next());
    delete(c);
    return true;
}

bool Predicate::covers(const Message & m) const {
    Predicate::Iterator * f = first();
    if (!f) return false;
    do {
	if (! f->covers(m)) {
	    delete(f);
	    return false;
	} 
    } while (f->next());
    delete(f);
    return true;
}

bool String::operator == (const String & x) const { 
    if (length() != x.length()) return false;
    const char * i = begin;
    const char * j = x.begin;
    while(i != end)
	if (*i++ != *j++) return false;
    return true;
}

bool String::operator < (const String & x) const { 
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

bool String::has_prefix (const String & x) const {
    if (length() < x.length()) return false;
    const char * i = begin;
    const char * j = x.begin;
    while(j != x.end)
	if (*i++ != *j++) return false;
    return true;
}

bool String::has_suffix (const String & x) const {
    if (length() < x.length()) return false;
    const char * i = begin;
    const char * j = x.begin;
    i += length() - x.length();
    while(j != x.end)
	if (*i++ != *j++) return false;
    return true;
}

bool String::has_substring (const String & x) const {
    unsigned int l = length();
    unsigned int xl = x.length();
    const char * b = begin;
    while (l >= xl) {
	const char * i = b;
	const char * j = x.begin;
	while(j != x.end)
	    if (*i++ != *j++) break;
	if (j == x.end) return true;
	++b;
	--l;
    }
    return false;
}

std::string & String::to_string(std::string & s) const {
    s.assign(begin, end);
    return s;
}

std::string String::to_string() const {
    return std::string(begin,end);
}

} // end namespace siena_impl

