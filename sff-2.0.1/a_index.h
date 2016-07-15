// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2002 University of Colorado
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
#ifndef A_INDEX_H_INCLUDED
#define A_INDEX_H_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "allocator.h"

namespace siena_impl {

class fwd_attribute;

class a_index_node;
/** @brief an attribute table in which attribute descriptors are
 *  accessed by name.
 *
 *  This table implements a map between attribute names and pointers
 *  to attribute descriptors.
 **/
class a_index {
public:
    a_index() { clear(); };

    fwd_attribute ** insert(const char *, const char *, batch_allocator &);
    const fwd_attribute * find(const char *, const char *) const;

    void clear();
    void consolidate();

private:
#ifdef WITH_A_INDEX_USING_TST
    a_index_node * roots[256];
#else
    a_index_node * root;
#endif
};

} // end namespace siena_impl

#endif
