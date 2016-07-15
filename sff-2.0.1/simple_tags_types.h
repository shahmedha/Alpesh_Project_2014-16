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
#ifndef SIMPLE_TAGS_TYPES_H_INCLUDED
#define SIMPLE_TAGS_TYPES_H_INCLUDED

#include <set>
#include <string>
#include <vector>

#include <siena/tags.h>

/** An implementation of siena::TagSet based on containers of the
 *  standard library.
 *
 *  This implementation uses a set of strings (i.e.,
 *  std::set<std::string>) as the storage for the tagset.
 */
class simple_tagset : public siena::TagSet {
public:
    class iterator : public siena::TagSet::Iterator {
    public:
	virtual bool next();

	virtual std::string to_string() const;
	virtual std::string & to_string(std::string & x) const;

	iterator(const std::set<std::string> * s, 
		 std::set<std::string>::const_iterator i);

	virtual ~iterator();

    private:
	const std::set<std::string> * ts;
	std::set<std::string>::const_iterator itr;
    };

    void add_tag(const std::string & s);
    virtual siena::TagSet::Iterator * first() const;

private:
    std::set<std::string> ts;
};

/** An implementation of siena::TagSetList based on containers of the
 *  standard library.
 *
 *  This implementation uses a vector of pointers to simple_tagset
 *  objects (i.e., vector<const simple_tagset *>) as the storage for
 *  the tagset.
 */
class simple_tagset_list: public siena::TagSetList {
public:
    class iterator : public siena::TagSetList::Iterator {
    public:
	virtual bool next();

	iterator(const std::vector<const simple_tagset *> * v, 
		 std::vector<const simple_tagset *>::const_iterator i);

 	virtual siena::TagSet::Iterator * first() const;

	virtual ~iterator();

    private:
	const std::vector<const simple_tagset *> * tsv;
	std::vector<const simple_tagset *>::const_iterator itr;
    };

    void add_tagset(const simple_tagset * s);
    virtual siena::TagSetList::Iterator * first() const;

    ~simple_tagset_list();

private:
    std::vector<const simple_tagset *> l;
};

#endif
