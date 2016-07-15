#!/bin/sh
#
#  This file is part of Siena, a wide-area event notification system.
#  See http://www.inf.usi.ch/carzaniga/siena/
#
#  Author: Antonio Carzaniga
#  See the file AUTHORS for full details. 
#
#  Copyright (C) 2005  Antonio Carzaniga
#
#  Siena is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  
#  Siena is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with Siena.  If not, see <http://www.gnu.org/licenses/>.
#
. $srcdir/config.sh
#
cleanup_files='bloom_encoding.out bloom_encoding.good'
#
$SFF -f b $srcdir/misc.inp > bloom_encoding.good
for algo in t s c v B Z X ; do
    if $SFF -f $algo < /dev/null; then
	test_description "Testing Bloom-encoding algorithm '$algo' against naive algorithm..."
	$SFF -f $algo $srcdir/misc.inp > bloom_encoding.out
	if diff bloom_encoding.out bloom_encoding.good; then
	    test_passed_continue
	else
	    test_failed 'see misc.inp, bloom_encoding.out, and bloom_encoding.good for detail'
	fi
    else
	test_description "Skipping Bloom-encoding algorithm '$algo' (not available)..."
	test_passed_continue
    fi
done
cleanup
exit 0
