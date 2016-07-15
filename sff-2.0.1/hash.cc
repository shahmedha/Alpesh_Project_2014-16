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
#include "hash.h"

namespace siena_impl {

// I need a prime that is close to a power of two.
//
// P = 2^31 - 1 = 2147483647 works, although I suppose I should
// check that that can be used as an unsigned integer, that is,
// sizeof(unsigned int) >= 4
//
// P = 2^13 - 1 = 8191 also works
// P = 2^17 - 1 = 131071 also works 
//
static const unsigned int P = 131071U; 

//
// INPUT: string = c[1]c[2]c[3] ... c[N]
// OUTPUT: c[1]*3^N + c[2]*3^(N-1) + ... + c[N-1]*3 + c[N] mod P
//
// generic poly-mod-P hash with a polynomial base X
//
unsigned int hash(unsigned int X, const char* b, const char *e) {
    register unsigned int h = X;
    for (; b != e; ++b) {
	h = (X*h + static_cast<unsigned char>(*b)) % P;
    }
    return h;
}

//
// INPUT: string = c[1]c[2]c[3] ... c[N]
// OUTPUT: c[1]*2^N + c[2]*2^(N-1) + ... + c[N-1]*2 + c[N] mod P
//
// poly-mod-P hash specialized for X=2
//
unsigned int hash(const char* b, const char *e) {
    register unsigned int h = 2;
    for (; b != e; ++b) {
	h = (h << 1) + h + static_cast<unsigned char>(*b);
	if (h >= P) h %= P;
    }
    return h;
}
#if 0
//
// quick hacks
//
unsigned int hash(int x) {
    return static_cast<unsigned int>(x);
}

template <typename T>
unsigned int hash_of_binary_rep(const T & x) {
    return hash(reinterpret_cast<const char *>(&x), 
		reinterpret_cast<const char *>(&x) + sizeof(T));
}

unsigned int hash(double x) {
    return hash_of_binary_rep(x);
}

unsigned int hash(float x) {
    return hash_of_binary_rep(x);
}
#endif

} // end namespace siena_impl

