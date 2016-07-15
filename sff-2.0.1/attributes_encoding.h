// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2003-2004 University of Colorado
//  Copyright (C) 2005 Antonio Carzaniga
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
#ifndef ATTRIBUTES_ENCODING_H_INCLUDED
#define ATTRIBUTES_ENCODING_H_INCLUDED

#include <sstream>
#include <string>

#include <siena/attributes.h>

/** \file bset_encoding.h
 *
 *  This file defines the functions that implement the encoding of
 *  attributes and constraints into strings, and the corresponding
 *  encoding functions for messages and filters.  All the functions
 *  are parameterized with a template parameter representing the
 *  underlying Bloom filter type (bloom_filter_T).  These functions
 *  require that bloom filter type implement the function:
 *
 *  add(const char * begin, const char * end)
 * 
 *  that adds a string to the set.
 **/
namespace siena_impl {

template <class bloom_filter_T>
void encode_equals(bloom_filter_T & b, const siena::Attribute * a) {
    std::string sbuf(a->name().begin, a->name().end);
    std::ostringstream s(sbuf);
    s << '=';
    switch(a->type()) {
    case siena::STRING: {
	s << 's';
	siena::String v = a->string_value();
	s << std::string(v.begin, v.end);
	break;
    }
    case siena::INT: {
	s << 'i' << a->int_value();
	break;
    }
    case siena::DOUBLE: {
	s << 'd' << a->double_value();
	break;
    }
    case siena::BOOL: {
	if (a->bool_value()) {
	    s << 'T';
	} else {
	    s << 'F';
	}
	break;
    }
    case siena::ANYTYPE:
	s <<'*';
	break;
    }
    b.add(sbuf.data(), sbuf.data() + sbuf.size());
}

//
// encode_equals encodes a constraint or an attribute as an
// *existence* category.
//
template <class bloom_filter_T>
void encode_exists(bloom_filter_T & b, const siena::Attribute * a) {
    std::string sbuf(a->name().begin, a->name().end);
    std::ostringstream s(sbuf);
    s << '*';
    switch(a->type()) {
    case siena::STRING: 
	s << 's';
	break;
    case siena::INT: 
    case siena::DOUBLE: 
	s << 'n';
	break;
    case siena::BOOL: 
	s << 'b';
	break;
    case siena::ANYTYPE:
    default:
	s << '*';
	break;
    }
    b.add(sbuf.data(), sbuf.data() + sbuf.size());
}

template <class bloom_filter_T>
void encode_constraint(bloom_filter_T & b, const siena::Constraint * c) {
    if (c->op() == siena::EQ && c->type() != siena::ANYTYPE) {
	encode_equals(b, c); 
    } else {
	encode_exists(b, c); 
    } 
}

template <class bloom_filter_T>
void encode_attribute(bloom_filter_T & b, const siena::Attribute * a) {
    if (a->type() != siena::ANYTYPE)
	encode_equals(b, a); 
    encode_exists(b, a); 
}

template <class bloom_filter_T>
void encode(bloom_filter_T & b, const siena::Filter * f) {
    siena::Filter::Iterator * fi = f->first();
    if (!fi) return;

    do {
	encode_constraint(b, fi); 
    } while (fi->next());
    delete(fi);
}

template <class bloom_filter_T>
void encode(bloom_filter_T & b, const siena::Message * m) {
    siena::Message::Iterator * mi  = m->first();
    if (!mi) return;

    do {
	encode_attribute(b, mi); 
    } while (mi->next());
    delete(mi);
}

} // end namespace siena_impl

#endif
