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
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef WITH_TIMERS
#include "timing.h"
#include "timers.h"

namespace siena_impl {

Timer ifconfig_timer;
Timer consolidate_timer;
Timer match_timer;
Timer bloom_encoding_timer;
Timer string_match_timer;
Timer forward_timer;

TimerStack sff_timer_stack;

}
#endif
