// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2001-2002 University of Colorado
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
#ifndef SIENA_FORWARDING_H_INCLUDED
#define SIENA_FORWARDING_H_INCLUDED

#include <siena/attributes.h>
#include <siena/tags.h>

/** \file forwarding.h 
 *
 *  This header file defines Siena's forwarding module interface.
 **/
namespace siena {

/** @brief Interface type.  
 *
 *  For simplicity, an interface is represented by a number.  This is
 *  an explicit design choice.  More information can be associated
 *  with an interface through a secondary data structure (a vector or
 *  anything else).
 **/
typedef unsigned int InterfaceId;

/** @brief hook for the output function for matching interfaces.
 *
 *  The matching function of a forwarding table doesn't actually
 *  produce an output.  Instead, it delegates the processing of
 *  matching interfaces to a specialized <em>match handler</em>.  This
 *  base class defines the interface of such a handler.  Users of the
 *  forwarding table must implement this interface and pass it to the
 *  matching function.
 *
 *  <p>The \link forwarding_messages.cc forwarding_messages\endlink
 *  example shows how to set up and use match handlers.  
 **/
class MatchHandler {
public:
    /** @brief virtual destructor */
    virtual ~MatchHandler() {};

    /** @brief output function.
     *
     *  This function is called within the \link AttributesFIB::match(const Message &, MatchHandler &) const matching function\endlink
     *  of the forwarding table.  The parameter is the interface
     *  identifier of the matching interface.  This function may
     *  explicitly cause the matching function to terminate by
     *  returning true. 
     **/
    virtual bool output(InterfaceId) = 0;
};

/** @brief Basic services of a forwarding table. 
 *
 *  A forwarding table implements the forwarding function of a
 *  specific routing scheme.  This class abstracts the essential
 *  services of a forwarding table that do not depend on the routing
 *  scheme.  These amount to little more than memory management.
 *
 *  Depending on the implementation, a forwarding table is typically a
 *  \em dictionary data structure, meaning that it is compiled once
 *  and then used for matching repeatedly.  The table can be compiled
 *  by adding associations between predicates and interfaces (or
 *  whatever is specific of the addressing and routing scheme) but can
 *  not be modified by removing or modifying individual associations.
 *  In order to modify individual associations, the forwarding table
 *  must be completely cleared, using the \link clear() clear\endlink
 *  or \link clear_recycle() clear_recycle \endlink methods, and built
 *  over again.
 *
 *  More specifically, the forwarding table operates in two modes.
 *  Initially the forwarding table is in <em>configuration mode</em>.
 *  While in configuration mode, the forwarding table can be
 *  configured by associating predicates to interfaces (e.g., using
 *  the \link siena::AttributesFIB::ifconfig() ifconfig\endlink
 *  method).  Once all the interfaces have been associated with their
 *  predicate, the forwarding table must be prepared for matching by
 *  calling the \link siena::FIB::consolidate() consolidate\endlink
 *  method.  The consolidate method switches the forwarding table to
 *  <em>matching mode</em>, which means that the forwarding table can
 *  be used to match messages.  The forwarding table, can then be
 *  cleared and restored to configuration mode with the \link
 *  siena::AttributesFIB::clear() clear\endlink or \link
 *  siena::AttributesFIB::clear_recycle() clear_recycle\endlink
 *  methods.
 *
 *  Once in matching mode, the forwarding table can match incoming
 *  messages with the \link siena::AttributesFIB::match()
 *  match\endlink method.  The forwarding table delegates the
 *  processing of matched notifications to a given \em handler.  The
 *  MatchHandler class defines the interface of a handler.
 * 
 *  The forwarding table manages its memory allocation through a
 *  dedicated block-allocator.  When cleared with clear(), the
 *  forwarding table releases all the previously allocated memory.
 *  clear_recycle() can be used to clear the forwarding table and
 *  recycle previously allocated memory.  In this latter case,
 *  previously allocated memory is saved for future use.
 **/
class FIB {
public:
    /** @brief Destroys the forwarding including all its internal data
     *  structures.
     **/
    virtual ~FIB() {};

    /** @brief Prepares the forwarding table for matching.
     *
     *  This function processes the forwarding table, packing some of
     *  its internal data structures and preparing them to be used to
     *  match events.  This function must therefore be called after
     *  all the necessary calls to \link siena::AttributesFIB::ifconfig() ifconfig\endlink
     *  and before matching messages with 
     *  \link siena::AttributesFIB::match(const Message &, MatchHandler &) const match()\endlink.
     *  <p>The forwarding table can be reset by calling 
     *  \link clear() clear\endlink or 
     *  \link clear_recycle() clear_recycle\endlink.
     *
     *  @see clear()
     *  @see recycle_clear()
     **/
    virtual void consolidate() {};

    /** @brief Clears the forwarding table.
     *
     *  This method removes all the associations from the forwarding
     *  table and releases allocated memory.  After a call to this
     *  method, the forwarding table is ready to be configured with
     *  \link siena::AttributesFIB::ifconfig()\endlink.
     *
     *  @see siena::AttributesFIB::ifconfig()
     *  @see consolidate()
     **/
    virtual void clear() = 0;

    /** @brief Clears the forwarding table.
     *
     *  This method removes all the associations from the forwarding
     *  table recycling the allocated memory. After a call to this
     *  method, the forwarding table is ready to be configured with
     *  \link siena::AttributesFIB::ifconfig()\endlink.
     *
     *  @see siena::AttributesFIB::ifconfig()
     *  @see consolidate()
     **/
    virtual void clear_recycle() = 0;

    /** @brief Memory allocated by the forwarding table.
     *
     *  returns the number of bytes of memory allocated by the
     *  forwarding table.  This value is always greater than or equal
     *  to the value returned by bytesize().
     **/
    virtual size_t allocated_bytesize() const = 0;

    /** @brief Memory used by the forwarding table.
     *
     *  returns the number of bytes of memory used by the forwarding
     *  table.  This value is always less than or equal to the value
     *  returned by allocated_bytesize().
     **/
    virtual size_t bytesize() const = 0;
};

/** @brief A forwarding table for the attribute-based, single-tree model. 
 *
 *  An AttributesFIB associates predicates to interfaces, as in a
 *  single-tree routing scheme.  An AttributesFIB uses the
 *  attribute-based data model, thus a predicate in an AttributesFIB
 *  is a disjunction of conjunctions of elementary constraints posed
 *  on the values of attributes in messages (or message descriptors).
 *  An interface is simply an identifier, corresponding to the
 *  link-level address of a neighbor.
 **/
class AttributesFIB : public FIB {
public:
    /** @brief Associates a predicate to an interface.
     *
     *  This is the method that constructs the forwarding table.  This
     *  method must be called \em once for each interface, after
     *  the forwarding table is constructed or after it has been
     *  cleared.  Using this method twice on the same interface
     *  without clearing the forwarding table has undefined effects.
     *
     *  @see consolidate()
     **/
    virtual void ifconfig(InterfaceId, const Predicate &) = 0;

    /** @brief Processes a message, calling the output() function on
     *	the given MatchHandler object for each matching interface.
     *
     *  Matches a message against the predicates stored in the
     *  forwarding table.  The result is processed through the
     *  MatchHandler passed as a parameter to this function.
     *
     *  Notice that the forwarding table must be consolidated by
     *  calling \link consolidate()\endlink before this function is
     *  called.
     *
     *  @see consolidate()
     **/
    virtual void match(const Message &, MatchHandler &) const = 0;
};

/** @brief A forwarding table for the tag-based, single-tree model. 
 *
 *  A TagsFIB associates predicates to interfaces as in a single-tree
 *  routing scheme.  A TagsFIB uses the tag-based model, thus a
 *  predicate in a TagsFIB is a lists of tagsets. A tagset is a set of
 *  tags representing a set of requested properties of a message.  An
 *  interface is simply an identifier, corresponding to the link-level
 *  address of a neighbor.
 **/
class TagsFIB : public FIB {
public:
    /** @brief Associates a predicate to an interface.
     *
     *  This is the method that constructs the forwarding table.  This
     *  method must be called \em once for each interface, after
     *  the forwarding table is constructed or after it has been
     *  cleared.  Using this method twice on the same interface
     *  without clearing the forwarding table has undefined effects.
     *
     *  @see consolidate()
     **/
    virtual void ifconfig(InterfaceId, const TagSetList &) = 0;

    /** @brief Processes a message, calling the output() function on
     *	the given MatchHandler object for each matching interface.
     *
     *  Matches a message against the predicates stored in the
     *  forwarding table.  The result is processed through the
     *  MatchHandler passed as a parameter to this function.
     *
     *  Notice that the forwarding table must be consolidated by
     *  calling \link consolidate()\endlink before this function is
     *  called.
     *
     *  @see consolidate()
     **/
    virtual void match(const TagSet &, MatchHandler &) const = 0;
};

} // end namespace siena

#endif
