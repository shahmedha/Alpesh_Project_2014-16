// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2013 University of Colorado
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
#ifndef SIENA_TAGS_H_INCLUDED
#define SIENA_TAGS_H_INCLUDED

#include <string>

/** \file tags.h 
 *
 *  This header file defines the basic types of the Siena data model
 *  defined on <em>tag sets</em>.  Within the tag-sets data model,
 *  data is described by sets of tags.  A request for some data is
 *  also defined by a set of tags, and would match data described by a
 *  superset of those tags.  In essence, this file defines messages
 *  and filters as sets of tags, and predicates as sets of filters.
 **/

namespace siena {

/** @brief interface of a generic <em>tag</em> in the Siena data model.
 *
 *  A <em>tag</em> is essentially a \em reference to the value of the
 *  tag, which is simply a string.  This very basic interface defines
 *  what amounts to an <em>immutable</em> tag.
 **/
class Tag {
public:
    /** @brief virtual destructor. */
    virtual ~Tag() {};

    /** @brief string value of this tag. 
     *
     *  @return this tag expressed as a standard string object.
     **/
    virtual std::string to_string() const = 0;

    /** @brief string value of this tag. 
     *
     *  Assigns to the string value of this tag to the given standard
     *  string object.
     **/
    virtual std::string	& to_string(std::string &) const = 0;
};

/** @brief interface of a generic <em>tag set</em> in the tag-based
 *  data model.
 *
 *  An <em>tag set</em> is what it says it is: a set of tags.  This
 *  basic interface defines an <em>immutable</em> tag set.
 *  Implementations of this interface may then have methods to add or
 *  remove tags.  The tags in a tag set can be accessed sequentially
 *  through a TagSet::Iterator.  The iteration order is implementation
 *  dependent.
 **/
class TagSet {
public:
    virtual ~TagSet() {};

    /** @brief iterator for a tag set.
     *
     *  An iterator in a tag set represents a tag \em directly, in the
     *  sense that it is itself a tag reference, and it can also be
     *  switched to refer to the next tag in the set.  The iteration
     *  order is implementation-dependent.
     *
     *  The following example shows how to iterate through a tag set:
     *
     *  @code
     *  #include <siena/tags.h>
     * 
     *  //... implementation of the siena/tags objects
     *  SimpleTagSet ts;
     *  // ...
     *  siena::TagSet::Iterator * i = ts.first();
     *  if (i != 0) {
     *      do {
     *          std::cout << i->to_string() << std::endl;
     *      } while(i->next());
     *      delete(i);
     *  }
     *  @endcode
     *
     *  Notice that iterators must be explicitly deallocated.
     **/
    class Iterator : public Tag {
    public:
	/** @brief moves this iterator to the next tag in the set.
	 *
	 *  @return \em true if the element pointed to by this
	 *  iterator \em before this call to next() is not the last
	 *  tag in its tag set.  In this case, after this call to
	 *  next(), this iterator will point to the next tag.  @return
	 *  \em false if the element pointed to by this iterator \em
	 *  before this call to next() is either the the last tag in
	 *  its tag set, or an invalid tag.  In this case, after this
	 *  call to next(), this iterator will point to an invalid
	 *  tag.
	 **/
	virtual bool next() = 0;

	/** @brief iterator destructor. */
	virtual ~Iterator() {};
    };

    /** @brief returns an iterator over this tag set. 
     *  
     *  The iterator returned by this method must define a complete
     *  iteration through the tag set.  The order of the iteration is
     *  implementation-dependent.
     *
     *  @return iterator pointing to the first tag in this tag set, or
     *  0 (\em NULL) if this tag set is empty.
     **/
    virtual Iterator * first() const = 0;
};

/** @brief interface of a generic \em predicate in the "tagset" Siena
 *  data model.
 *
 *  A \em predicate is essentially a collection of tag sets, thus the
 *  name TagSetList.  Each tag set \em F in a predicate represents a
 *  logical \em filter that matches a message described by another
 *  tagset \em M when \em M is a superset of \em F.  Thus a message
 *  defined by a tag set \em M matches a predicate \em P if and only
 *  if \em P contains at least one tag set \em F such that \em M is a
 *  superset of \em F.  The tag sets in a predicate can be accessed
 *  sequentially through a TagSetList::Iterator.
 *  
 *  The following code exemplifies the structure and access mode for a
 *  TagSetList:
 *
 *  @code
 *  #include <siena/tags.h>
 * 
 *  class SimpleTagSetList : public siena::TagSetList {
 *      //... 
 *  };
 *
 *  SimpleTagSetList tsl;
 *  // ... here we
 *
 *  //
 *  // we now print the content of tsl
 *  //
 *  siena::TagSetList::Iterator * ts = tsl.first();
 *  if (ts == 0) {
 *      cout << "no tag sets" endl;
 *  } else {
 *      cout << "Tag set:";
 *      siena::TagSet::Iterator * ts->first();
 *      if (i != 0) {
 *          do {
 *              std::cout << " " << i->to_string();
 *          } while(i->next());
 *          delete(i);
 *          cout << std::endl;
 *      }
 *      delete(ts);
 *  }
 *  @endcode
 *
 *  Notice that iterators must be explicitly deallocated.
 **/
class TagSetList {
public:
    /** @brief virtual destructor */
    virtual ~TagSetList() {};

    /** @brief interface of a generic reference to a tagset in a tagset list.
     *
     *  Provides sequential access to the individual tag sets in a
     *  tagset list.  The iterator provides direct access to the tag
     *  set in the sense that it itself implements the TagSet interface.
     **/
    class Iterator : public TagSet {
    public:
	/** @brief moves this iterator to the next tag set in its predicate.
	 *
	 *  @return \em true if the tag set pointed to by this
	 *  iterator \em before this call to next() is not the last
	 *  one in its predicate.  In this case, after this call to
	 *  next(), this iterator will point to the next tag set.
	 * 
	 *  @return \em false if this iterator points to the last tag
	 *  set or to the end of the sequence.  In this case, after
	 *  this call to next(), this iterator will point to an
	 *  invalid tag set representing the end of the sequence.
	 **/
	virtual bool next() = 0;
	/** @brief destructor */
	virtual ~Iterator() {};
    };

    /** @brief returns an iterator over this predicate. 
     *  
     *  @return iterator pointing to the first tag set in this
     *  predicate, or 0 (\em NULL) if this predicate is empty.
     **/
    virtual Iterator * first() const = 0;
};

} // end namespace siena

#endif
