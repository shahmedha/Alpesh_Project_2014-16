//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//  
//  Author: Antonio Carzaniga <firstname.lastname@usi.ch>
//  See the file AUTHORS for full details. 
//  
//  Copyright (C) 2005 Antonio Carzaniga
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
#ifndef SIENA_TIMERS_H
#define SIENA_TIMERS_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef WITH_TIMERS
#include "timing.h"
#include <iostream>

namespace siena_impl {

    extern Timer ifconfig_timer;
    extern Timer consolidate_timer;
    extern Timer match_timer;
    extern Timer bloom_encoding_timer;
    extern Timer string_match_timer;
    extern Timer forward_timer;

    class TimerStack {
    public:
	/** maximum depth for the timer stack.
	 */
	static const unsigned int MAX_DEPTH = 256;

    private:
	Timer * tstack[MAX_DEPTH];
	unsigned int depth;

    public:
	class TimerStackOverflow : public std::exception {
    public:
	virtual const char* what() const throw() { 
	    return "timer stack overflow";
	}
    };

    class TimerStackUnderflow : public std::exception {
    public:
	virtual const char* what() const throw() { 
	    return "timer stack underflow";
	}
    };

    /** pushes the given timer on this stack of timers.
     *
     *  This class maintains a stack of timers to allow per-procedure
     *  or per-block accounting.  The semantics of the timer stack is
     *  as follows: when a timer T is pushed onto the stack, T is
     *  started.  At the same time, the timer T' that was on top of
     *  the stack (if any) is stopped.  When a timer T is popped from
     *  the stack, the T is stopped, and at the same time the timer T'
     *  that emerges at the top of the stack (if any) is started.
     *  This semantics is illustrated by the following example:
     * 
     *  <code>
     *  TimerStack S;
     *  Timer timer_x;
     *  Timer timer_y;
     *
     *  //...
     *  S.push(timer_x);
     *  //... some code that executes for 3 seconds
     *  S.push(timer_y);
     *  //... some code that executes for 10 seconds
     *  S.pop(timer_y);
     *  //... some code that executes for 2 seconds
     *  S.pop(timer_x);
     *  std::cout << "Tx=" << (timer_x.read() / 1000000) << std::endl;
     *  std::cout << "Ty=" << (timer_y.read() / 1000000) << std::endl;
     *  </code>
     *
     *  the example should output something like:
     *  <code>
     *  Tx=5
     *  Ty=10
     *  </code>
     */
    TimerStack(): depth(0) {};

    void push(Timer * x) throw (TimerStackOverflow) {
	if (depth < MAX_DEPTH) {
	    if (depth == 0) {
		tstack[0] = x;
		++depth;
		x->start();
	    } else if (tstack[depth - 1] != x) {
		tstack[depth - 1]->stop();
		tstack[depth] = x;
		++depth;
		x->start();
	    } else {
		tstack[depth] = x;
		++depth;
	    }
	} else {
	    throw TimerStackOverflow();
	}
    }

    void pop() throw (TimerStackUnderflow) {
	if (depth > 0) {
	    --depth;
	    if (depth == 0) {
		tstack[0]->stop();
	    } else if (tstack[depth] != tstack[depth - 1]) {
		tstack[depth]->stop();
		tstack[depth - 1]->start();
	    }
	} else {
	    throw TimerStackUnderflow();
	}
    }
    };

    extern TimerStack sff_timer_stack;

#if DEBUG_OUTPUT
#define TIMER_PUSH(x) {							\
	std::cout << __FILE__ << ':' << __LINE__ << '(' << std::flush;	\
	siena_impl::sff_timer_stack.push(&(x)); }

#define TIMER_POP(x) {							\
	std::cout << __FILE__ << ':' << __LINE__ << ')' << std::flush;	\
	siena_impl::sff_timer_stack.pop(); }
#else
#define TIMER_PUSH(x) { siena_impl::sff_timer_stack.push(&(x)); }
#define TIMER_POP(x) { siena_impl::sff_timer_stack.pop(); }
#endif
}
#else // no timers
#define TIMER_PUSH(x)
#define TIMER_POP()
#endif

#endif
