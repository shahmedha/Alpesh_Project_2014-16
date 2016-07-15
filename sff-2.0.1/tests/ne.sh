#!/bin/sh
#
#  This file is part of Siena, a wide-area event notification system.
#  See http://www.inf.usi.ch/carzaniga/siena/
#
#  Author: Antonio Carzaniga
#  See the file AUTHORS for full details. 
#
#  Copyright (C) 1998-2003 University of Colorado
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
cleanup_files=''
#
set $srcdir/ne*.inp
num=$#
n=1
for in
do
  expected=`echo $in | sed 's/inp$/good/'`
  out=`basename $in | sed 's/inp$/out/'`
  cleanup_files="$out $cleanup_files"
  test_description "Testing not-equals constraints with integers ($n/$num)..."
  $SFF < $in > $out
  n=`expr $n + 1`
  if diff $out $expected
  then
      test_passed_continue
  else
      test_failed "see $in, $out, $expected, and test.log for detail"
  fi
done
test_description "Tested not-equals constraints with integers..."
test_passed
