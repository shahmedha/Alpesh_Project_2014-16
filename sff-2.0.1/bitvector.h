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
#ifndef SIENA_BITVECTOR_H
#define SIENA_BITVECTOR_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <cstddef> /* for size_t */
#include <climits> /* for CHAR_BIT*/
#include <cstring> /* for memset */

#include "allocator.h"

namespace siena_impl {

typedef unsigned long bv_block;

#define BV_BLOCK_ONE (1UL)
#define BV_BLOCK_ZERO (0UL)
#define BV_BITS_PER_BLOCK (sizeof(bv_block) * CHAR_BIT)
#define BV_BYTES_PER_BLOCK (BV_BITS_PER_BLOCK / 8)

#define BV_BLOCKS_NEEDED(bits) ((bits) / BV_BITS_PER_BLOCK + ((bits) % BV_BITS_PER_BLOCK == 0 ? 0 : 1))
#define BV_POS_IN_BLOCK(pos) (pos % BV_BITS_PER_BLOCK)
#define BV_BLOCK_INDEX(pos) (pos / BV_BITS_PER_BLOCK)
#define BV_BLOCK_COVERS(a,b) ((a & b) == b)

#ifdef HAVE_BUILTIN_POPCOUNT
inline size_t popcount(bv_block b) {
    return __builtin_popcountl(b);
}
#else
// popcount_4 from Wikipedia (http://en.wikipedia.org/wiki/Hamming_weight) 
inline size_t popcount(bv_block x) {
    size_t count;
    for (count=0; x; ++count)
        x &= x-1;
    return count;
}
#endif

template <size_t BlockSize>
class bitvector_blocks {
private:
    bv_block v[BlockSize];

public:
    bitvector_blocks() {
	memset(v, 0, sizeof(v));
    }

    void clear() {
	memset(v, 0, sizeof(v));
    }

    void set(size_t pos) {
	v[BV_BLOCK_INDEX(pos)] |= (BV_BLOCK_ONE << BV_POS_IN_BLOCK(pos));
    }

    bool test_and_set(size_t pos) {
	size_t i = BV_BLOCK_INDEX(pos);

	bv_block mask = (BV_BLOCK_ONE << BV_POS_IN_BLOCK(pos));
	if ((v[i] & mask) == 0) {
	    v[i] |= mask;
	    return false;
	} else 
	    return true;
    }

    bool test(size_t pos) const {
	return (v[BV_BLOCK_INDEX(pos)] & (BV_BLOCK_ONE << BV_POS_IN_BLOCK(pos)));
    }

    bool operator[](size_t pos) const {
	return (v[BV_BLOCK_INDEX(pos)] & (BV_BLOCK_ONE << BV_POS_IN_BLOCK(pos)));
    }

    bool covers(const bitvector_blocks<BlockSize> & b) const {
	for(size_t i = 0; i < BlockSize; ++i)
	    if ((v[i] & b.v[i]) != b.v[i])
		return false;
	return true;
    }

    size_t count() const {
	size_t res = 0;
	for(size_t i = 0; i < BlockSize; ++i)
	    res += popcount(v[i]);
	return res;
    }

    // 8-bit blocks (a.k.a., bytes)
    //
    typedef unsigned char byte_t;

    // number of 8-bit blocks
    //
    static const size_t B8Size = BV_BYTES_PER_BLOCK * BlockSize;

    /** \brief 8-bit block (byte) at position pos
     */
    byte_t bv8(size_t pos) const {
	return (v[BV_BLOCK_INDEX(pos * 8)] >> BV_POS_IN_BLOCK(pos * 8)) & 0xff;
    }
};

template <>
class bitvector_blocks<1> {
private:
    bv_block v;

public:
    bitvector_blocks() : v(0) { }

    void set(size_t pos) {
	v |= (BV_BLOCK_ONE << BV_POS_IN_BLOCK(pos));
    }

    bool test_and_set(size_t pos) {
	bv_block mask = (BV_BLOCK_ONE << BV_POS_IN_BLOCK(pos));
	if ((v & mask) == 0) {
	    v |= mask;
	    return false;
	} else 
	    return true;
    }

    bool test(size_t pos) const {
	return (v & (BV_BLOCK_ONE << BV_POS_IN_BLOCK(pos)));
    }

    bool operator[](size_t pos) const {
	return (v & (BV_BLOCK_ONE << BV_POS_IN_BLOCK(pos)));
    }

    bool covers(const bitvector_blocks<1> & b) const {
	return ((v & b.v) == b.v);
    }

    void clear() {
	v = 0;
    }

    size_t count() const {
	return popcount(v);
    }

    // 8-bit blocks (a.k.a., bytes)
    //
    typedef unsigned char byte_t;

    // number of 8-bit blocks
    //
    static const size_t B8Size = BV_BYTES_PER_BLOCK;

    /** \brief 8-bit block (byte) at position pos
     */
    byte_t bv8(size_t pos) const {
	return (v >> BV_POS_IN_BLOCK(pos * 8)) & 0xff;
    }
};

template <size_t M> 
class fixed_bitvector : public bitvector_blocks<BV_BLOCKS_NEEDED(M)> {
    typedef bitvector_blocks<BV_BLOCKS_NEEDED(M)> base_blocks;

public:
    fixed_bitvector(): base_blocks() {};
    static const size_t WIDTH = M;
};

class bitvector;
class ibitvector;

/** fixed-size bit vector.
 *
 *  This bitvector is implemented by a single array.
 **/
class bitvector {
public:
    /** constructs a bitvector capable of holding <em>size</em> bits
     **/
    bitvector(size_t size, bool v = false);
    bitvector(const bitvector & bv);
    ~bitvector();
    
    bool			test(size_t pos) const;
    bool			operator[](size_t pos) const;
    bool			set(size_t pos);
    bool			clear(size_t pos);

    void			set(const ibitvector & x);

    void			clear();
    /** number of bits set to 1 **/
    size_t			get_count() const;
    /** total bit capacity **/
    size_t			get_size() const;

    bool			operator < (const bitvector & x) const;
    bool			operator ==(const bitvector & x) const;
    bitvector &			operator = (const bitvector & x);
    bitvector &			operator &= (const bitvector & x);

protected:
    bv_block *			elements; 
    const size_t		size;		// number of bits
    size_t			count;		// number of bits set to 1

    static size_t		set(bv_block * x, 
				    bv_block * xe, 
				    const bv_block * y, 
				    const bv_block *ye);

    static size_t atom_size(size_t s) { 
	return BV_BLOCKS_NEEDED(s);
    }

    size_t element_size() { 
	return BV_BLOCKS_NEEDED(size);
    }

    static size_t atom_position(size_t s) { 
	return BV_BLOCK_INDEX(s);
    }

    static size_t bit_position(size_t s) { 
	return BV_POS_IN_BLOCK(s);
    }

    bitvector(bv_block *, size_t size, bool v = false);
    bitvector(bv_block *, const bitvector & bv);
};

/** utility wrapper for bitvector.  The only difference w.r.t. bitvector
 *  is that FABitvector is constructed with an external allocator.  We
 *  use it in XDDNodeMap.
 **/
class FABitvector: public bitvector {
public:
    FABitvector(batch_allocator & mem, const bitvector & bv) 
	: bitvector(new (mem) bv_block[atom_size(bv.get_size())], bv) {}
    FABitvector(batch_allocator & mem, size_t size, bool v = false) 
	: bitvector(new (mem) bv_block[atom_size(size)], size, v) {}
    ~FABitvector() {}
};

/** variable-size bit vector.
 *
 *  This bit vector is implemented by a multi-index structure.
 **/
class ibitvector {
public:
    ibitvector();
    
    bool			test(size_t pos) const;
    bool			set(size_t pos, batch_allocator &);

    void			clear();
    size_t			get_count() const;
    size_t			get_size() const;

private:
    struct index;
    struct block;

    static const unsigned int	block_size = 16;
    static const unsigned int	index_size = 16;

    union index_or_block {
	block *		b;
	index *		i;
    };

    struct block {
	index *		up;
	bv_block	elements[block_size];
	
	block(index *);
    };

    struct index {
	index *		up;
	index_or_block	down[index_size];

	index(index *);
    };

    class iterator {
    public:
	iterator(const block &);

	size_t		element_address() const;
	const block *	next_block();

    private:
	index *		bi;
	size_t		addr;
	unsigned char	level;
    };

    size_t count;
    size_t size;
    block  first_block;

    friend class bitvector;
};

} // end namespace siena_impl

#include "bitvector.icc"

#endif
