// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2002-2003 University of Colorado
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
#include <siena/forwarding.h>
#include "allocator.h"

#include "bool_index.h"

namespace siena_impl {

fwd_constraint * BoolIndex::add_eq(bool v, batch_allocator & ftmemory) {
    if (v) {
	if (t == 0) t = new (ftmemory)fwd_constraint();
	return t;
    } else {
	if (f == 0) f = new (ftmemory)fwd_constraint();
	return f;
    }
}

fwd_constraint * BoolIndex::add_ne(bool v, batch_allocator & ftmemory) {
    return add_eq(!v, ftmemory);
}

fwd_constraint * BoolIndex::add_any(batch_allocator & ftmemory) {
    if (any_value == 0)
	any_value = new (ftmemory)fwd_constraint();
    return any_value;
}

}
