// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2003 University of Colorado
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
#ifndef SIENA_HASH_H_INCLUDED
#define SIENA_HASH_H_INCLUDED

namespace siena_impl {

// first we define a couple of simple hash functions for strings

/** computes the hash of the string that starts at begin pointer and
 *  ends right before end pointer.
 */
extern unsigned int hash(const char* begin, const char *end);
#if 0
extern unsigned int hash(int i);
extern unsigned int hash(double d);
extern unsigned int hash(float f);
#endif

extern unsigned int hash(unsigned int X, const char* begin, const char *end);

} // end namespace siena_impl

#endif
