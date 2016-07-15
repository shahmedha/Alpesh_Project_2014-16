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
// $Id: types.cc,v 1.6 2010-03-11 15:16:43 carzanig Exp $
//
#include "siena/types.h"

namespace siena {

    bool constraint::covers(const attribute & a) const throw() {
	switch(op()) {
	case eq_id: 
	    switch(type()) {
	    case string_id: return (a.type() == string_id && 
				    a.string_value() == string_value());
	    case int_id: return (a.type() == int_id && 
				 a.int_value() == int_value());
	    case double_id: return (a.type() == double_id &&
				    a.double_value() == double_value());
	    case bool_id: return (a.type() == bool_id &&
				  a.bool_value() == bool_value());
	    case anytype_id: return true;
	    }
	case lt_id:
	    switch(type()) {
	    case string_id: return (a.type() == string_id && 
				    a.string_value() < string_value());
	    case int_id: return (a.type() == int_id && 
				 a.int_value() < int_value());
	    case double_id: return (a.type() == double_id &&
				    a.double_value() < double_value());
	    case bool_id: return (a.type() == bool_id &&
				  a.bool_value() < bool_value());
	    case anytype_id: return true;
	    }
	case gt_id:
	    switch(type()) {
	    case string_id: return (a.type() == string_id && 
				    a.string_value() > string_value());
	    case int_id: return (a.type() == int_id && 
				 a.int_value() > int_value());
	    case double_id: return (a.type() == double_id &&
				    a.double_value() > double_value());
	    case bool_id: return (a.type() == bool_id &&
				  a.bool_value() > bool_value());
	    case anytype_id: return true;
	    }
	case sf_id: 
	    switch(type()) {
	    case string_id: return (a.type() == string_id && 
				    a.string_value().has_suffix(string_value()));
	    case anytype_id: return true;
	    case int_id:
	    case double_id:
	    case bool_id:
		return false;
	    }
	case pf_id: 
	    switch(type()) {
	    case string_id: return (a.type() == string_id && 
				    a.string_value().has_prefix(string_value()));
	    case anytype_id: return true;
	    case int_id:
	    case double_id:
	    case bool_id:
		return false;
	    }
	case ss_id: 
	    switch(type()) {
	    case string_id: return (a.type() == string_id && 
				    a.string_value().has_substring(string_value()));
	    case anytype_id: return true;
	    case int_id:
	    case double_id:
	    case bool_id:
		return false;
	    }
	case re_id: 
	    switch(type()) {
	    case string_id: return false;
	    case anytype_id: return true;
	    case int_id:
	    case double_id:
	    case bool_id:
		return false;
	    }
	case any_id: 
	    switch(type()) {
	    case string_id: return (a.type() == string_id);
	    case int_id: return (a.type() == int_id);
	    case double_id: return (a.type() == double_id);
	    case bool_id: return (a.type() == bool_id);
	    case anytype_id: return true;
	    }
	case ne_id: 
	    switch(type()) {
	    case string_id: return (a.type() == string_id && 
				    a.string_value() != string_value());
	    case int_id: return (a.type() == int_id && 
				 a.int_value() != int_value());
	    case double_id: return (a.type() == double_id &&
				    a.double_value() != double_value());
	    case bool_id: return (a.type() == bool_id &&
				  a.bool_value() != bool_value());
	    case anytype_id: return true;
	    }
	}
	return false;
    }

    bool filter::covers(const message & m) const throw() {
	filter::iterator * c = first();
	if (!c) return false;
	do {
	    attribute * a = m.find(c->name());
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

    bool predicate::covers(const message & m) const throw() {
	predicate::iterator * f = first();
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

    /** @brief prefix
     **/
    bool string_t::has_prefix (const string_t & x) const throw() {
	if (length() < x.length()) return false;
	const char * i = begin;
	const char * j = x.begin;
	while(j != x.end)
	    if (*i++ != *j++) return false;
	return true;
    }

    /** @brief suffix
     **/
    bool string_t::has_suffix (const string_t & x) const throw() {
	if (length() < x.length()) return false;
	const char * i = begin;
	const char * j = x.begin;
	i += length() - x.length();
	while(j != x.end)
	    if (*i++ != *j++) return false;
	return true;
    }

    /** @brief substring
     **/
    bool string_t::has_substring (const string_t & x) const throw() {
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

} // end namespace siena

