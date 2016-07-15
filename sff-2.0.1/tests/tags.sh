#!/bin/sh
#
#  This file is part of Siena, a wide-area event notification system.
#  See http://www.inf.usi.ch/carzaniga/siena/
#
#  Author: Antonio Carzaniga
#  See the file AUTHORS for full details. 
#
#  Copyright (C) 2013  Antonio Carzaniga
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
cleanup_files='tags_sanitized.out tags_sanitized.good'
#
for algo in T Tt ; do
    if $SFF -f $algo < /dev/null; then
	test_description "Testing TagsTable algorithm '$algo'..."
	$SFF -f $algo $srcdir/tags_sanitized.inp > tags_sanitized.out
	if diff tags_sanitized.out $srcdir/tags_sanitized.good; then
	    test_passed_continue
	else
	    test_failed 'see tags_sanitized.inp, tags_sanitized.out, and tags_sanitized.good for detail'
	fi
    else
	test_description "Skipping algorithm '$algo' (not available)..."
	test_passed_continue
    fi
done
cleanup
exit 0
