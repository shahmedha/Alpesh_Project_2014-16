// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2003-2004 University of Colorado
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
#ifndef BLOOM_FILTER_H_INCLUDED
#define BLOOM_FILTER_H_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "bitvector.h"
#include "hash.h"

/** \file bloom_filter.h 
 *
 *  This file defines the Bloom filter types.
 **/
namespace siena_impl {

template <unsigned M = CONFIG_BLOOM_FILTER_SIZE, unsigned K = CONFIG_BLOOM_FILTER_K> 
class bloom_filter : public fixed_bitvector<M> {
    typedef fixed_bitvector<M> base_bit_vector;

public:
    static const unsigned int WIDTH = M;

    void add(const char * begin, const char * end) {
	for(unsigned int k = 2; k < K + 2; ++k)
	    base_bit_vector::set(hash(k, begin, end) % M);
    }

    void add(const std::string & s) {
	for(unsigned int k = 2; k < K + 2; ++k)
	    base_bit_vector::set(hash(k, s.data(), s.data() + s.size()) % M);
    }

    bool contains(const char * begin, const char * end) const {
	for(unsigned int k = 2; k < K + 2; ++k)
	    if (! base_bit_vector::test(hash(k, begin, end) % M))
		return false;
	return true;
    }
};

template  <unsigned M = CONFIG_BLOOM_FILTER_SIZE, unsigned K = CONFIG_BLOOM_FILTER_K, 
	   class base_bit_setT = fixed_bitvector<M> > 
class bloom_filter_wrapper {
    typedef base_bit_setT base_bit_vector;

    base_bit_setT & bit_set;

public:
    bloom_filter_wrapper(base_bit_setT & s): bit_set(s) {}

    static const unsigned int WIDTH = M;

    void add(const char * begin, const char * end) {
	for(unsigned int k = 2; k < K + 2; ++k)
	    bit_set.set(hash(k, begin, end) % M);
    }

    void add(const std::string & s) {
	for(unsigned int k = 2; k < K + 2; ++k)
	    bit_set.set(hash(k, s.data(), s.data() + s.size()) % M);
    }

    bool contains(const char * begin, const char * end) const {
	for(unsigned int k = 2; k < K + 2; ++k)
	    if (! bit_set.test(hash(k, begin, end) % M))
		return false;
	return true;
    }
};

} // end namespace siena_impl

#endif
