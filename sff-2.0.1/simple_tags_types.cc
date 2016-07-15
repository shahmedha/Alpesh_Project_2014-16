// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2013 Antonio Carzaniga
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
#include <set>
#include <string>
#include <vector>

#include <siena/tags.h>

#include "simple_tags_types.h"

bool simple_tagset::iterator::next() {
    return (++itr != ts->end());
};

std::string simple_tagset::iterator::to_string() const {
    return *itr;
}

std::string & simple_tagset::iterator::to_string(std::string & x) const {
    x = *itr;
    return x;
}

simple_tagset::iterator::iterator(const std::set<std::string> * s, 
				  std::set<std::string>::const_iterator i)
    : ts(s), itr(i) {};

simple_tagset::iterator::~iterator() {};

siena::TagSet::Iterator * simple_tagset::first() const {
    std::set<std::string>::const_iterator b = ts.begin();
    if (b == ts.end()) {
	return 0;
    } else {
	return new simple_tagset::iterator(&ts,b);
    }
}

void simple_tagset::add_tag(const std::string & s) {
    ts.insert(s);
}

bool simple_tagset_list::iterator::next() {
    return (++itr != tsv->end());	    
};

simple_tagset_list::iterator::~iterator() {};

simple_tagset_list::iterator::iterator(const std::vector<const simple_tagset *> * v, 
				       std::vector<const simple_tagset *>::const_iterator i)
    : tsv(v), itr(i) {};

siena::TagSet::Iterator * simple_tagset_list::iterator::first() const {
    return (*itr)->first();
}

void simple_tagset_list::add_tagset(const simple_tagset * s) {
    l.push_back(s);
}

siena::TagSetList::Iterator * simple_tagset_list::first() const {
    std::vector<const simple_tagset *>::const_iterator b = l.begin();
    if (b == l.end()) {
	return 0;
    } else {
	return new simple_tagset_list::iterator(&l,b);
    }
}

simple_tagset_list::~simple_tagset_list() {
    for (std::vector<const simple_tagset *>::const_iterator i = l.begin();
	 i != l.end(); ++i)
	if(*i) delete(*i);
}
