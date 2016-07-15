// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2001-2003 University of Colorado
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
#ifndef SIENA_FWDTABLE_H_INCLUDED
#define SIENA_FWDTABLE_H_INCLUDED

#include <siena/forwarding.h>

namespace siena {

/** @brief implementation of a forwarding table based on an improved
 * "counting" algorithm.
 *
 *  This class implements the index structure and matching algorithm
 *  described in <b>"Forwarding in a Content-Based Network"</b>,
 *  <i>Proceedings of ACM SIGCOMM 2003</i>.
 *  p.&nbsp;163-174. Karlsruhe, Germany.  August, 2003.
 *
 *  In addition, this algorithm includes a number of improvements.
 *  The most notable one uses a fast containment check between the set
 *  of attribute names in the message and the constraint names of a
 *  candidate filter.  If the check is negative, that is, if the set
 *  of attribute names is not a superset of the set of names of a
 *  candidate filter, then the algorithm does not even bother to
 *  "count" the number of matched constraints for that filter.  The
 *  test is based on a representation of the set of names consisting
 *  of a Bloom filter.
 * 
 *  Another improvement uses static counters to implement a faster but
 *  non-reentrant counting algorithm.  This variant can be selected
 *  with the <code>--with-non-reentrant-counters</code> option at
 *  configure-time.
 */
class FwdTable : public AttributesFIB {
public:
    /** @brief Determines the number of pre-processing rounds applied to
     *	every message.
     **/
    virtual void set_preprocess_rounds(unsigned int) = 0;

    /** @brief Returns the current number of pre-processing rounds.
     **/
    virtual unsigned int get_preprocess_rounds() const = 0;

    /** @brief Creates a FwdTable object.
     **/
    static FwdTable * create();
};

} // end namespace siena

#endif
