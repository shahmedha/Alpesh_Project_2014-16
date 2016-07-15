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
#ifndef BOOL_INDEX_H
#define BOOL_INDEX_H

#include <siena/forwarding.h>
#include "allocator.h"

#include "fwd_table.h"

namespace siena_impl {

/** index for <em>equals</em> constraints on boolean attributes **/
class BoolIndex {
public:
    BoolIndex();

    fwd_constraint *		add_any(batch_allocator &);
    fwd_constraint *		add_ne(bool v, batch_allocator &);
    fwd_constraint *		add_eq(bool v, batch_allocator &);
    bool			match(bool v, c_processor & p) const;

private:
    fwd_constraint * any_value;
    fwd_constraint * t;
    fwd_constraint * f;
};

inline BoolIndex::BoolIndex() 
    : any_value(0), t(0), f(0) {}

inline bool BoolIndex::match(bool v, c_processor & p) const {
    if (any_value != 0) 
	if (p.process_constraint(any_value))
	    return true;
    if (v) {
	if (t != 0) return p.process_constraint(t);
    } else {
	if (f != 0) return p.process_constraint(f);
    }
    return false;
}

} // end namespace siena_impl

#endif
