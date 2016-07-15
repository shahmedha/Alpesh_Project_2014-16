// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Paolo Bonzini <bonzini@gnu.org>
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2005 Paolo Bonzini <bonzini@gnu.org>
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
#ifndef RANGE_SET_H
#define RANGE_SET_H

#include <cstdlib> // for strtol

#include "allocator.h"

using namespace std;

template <class T>
class range_set {
public:
    range_set() {
	root = 0;
    };

    range_set(const char *s) {
	root = 0;
	do {
	    int first = strtol(s, const_cast<char **>(&s), 10), last = first;
	    if (*s == '-') {
		s++;
	        last = strtol(s, const_cast<char **>(&s), 10);
	    }
	    add (first, last);
	} while (*s++);
    }

    ~range_set() {
	if (root) {
	    root->delete_children (true, true);
	    delete root;
	}
    }

    bool operator[] (T key) {
	node *p = root;
	while (p) {
	    if (key < p->first)
		p = p->left;
	    else if (key > p->last)
		p = p->right;
	    else
		return true;
	}
	return false;
    }

    void add (T key) {
	add (key, key);
    }

    void add (T first, T last) {
	node *p, **pp = &root;
	while ((p = *pp) != 0) {
	    if (last < p->first)
		pp = &(p->left);
	    else if (first > p->last)
		pp = &(p->right);
	    else {
		// Extend the current node to include FIRST and LAST.
		// The children are removed if they include some of the
		// values in the range.
	        if (first < p->first)
		    p->extend_first (first);
	        if (last > p->last)
		    p->extend_last (last);
		return;
	    }
	}
	*pp = new node (first, last);
    }

private:
    struct node {
	node(T f, T l) 
	    : first(f), last(l), left(0), right(0) {};
	T first;
	T last;
	node *left;
	node *right;
	bool includes (T value) {
	    return value >= first && value <= last;
	}
	void extend_first (T new_first) {
	    first = new_first;
	    while (left && left->last >= first) {
		if (first > left->first)
		    first = left->first;

		// Remove the left->right branch, all the nodes there
		// surely overlap with THIS.  left->left becomes the
		// new left branch, and we recurse.
		node *new_left = left->left;
		left->delete_children (false, true);
		delete left;
		left = new_left;
	    }
	}

	void extend_last (T new_last) {
	    last = new_last;
	    while (right && right->first <= last) {
		if (last < right->last)
		    last = right->last;

		// Remove the right->left branch, all the nodes there
		// surely overlap with THIS.  right->right becomes the
		// new right branch, and we recurse.
		node *new_right = right->right;
		right->delete_children (true, false);
		delete right;
		right = new_right;
	    }
	}

	// Delete one or both the branches of THIS.
	void delete_children (bool del_left, bool del_right)
	{
	    if (del_left && left) {
		left->delete_children (true, true);
		delete left;
		left = 0;
	    }
	    if (del_right && right) {
		right->delete_children (true, true);
		delete right;
		right = 0;
	    }
	}
    };

    node *root;
};

#endif
