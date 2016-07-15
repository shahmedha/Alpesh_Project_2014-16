# -*- shell-script -*-
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
: ${SFF:=../sff}
: ${logfile:=`basename $0 | sed -e 's/\.sh$//' -e 's/$/\.log/'`}
#
# I/O redirections
#
exec 3>&1
exec 1>$logfile 2>&1
#
# borrowed from autoconf
#
case `echo "testing\c"; echo 1,2,3`,`echo -n testing; echo 1,2,3` in
  *c*,-n*) ECHO_N= ECHO_C='
' ECHO_T='	' ;;
  *c*,*  ) ECHO_N=-n ECHO_C= ECHO_T= ;;
  *)       ECHO_N= ECHO_C='\c' ECHO_T= ;;
esac
# Be Bourne compatible
if test -n "${ZSH_VERSION+set}" && (emulate sh) >/dev/null 2>&1; then
  emulate sh
  NULLCMD=:
elif test -n "${BASH_VERSION+set}" && (set -o posix) >/dev/null 2>&1; then
  set -o posix
fi
#
# common functions
#
test_description () {
    test -z "$QUIET" && echo ${ECHO_N} "$*${ECHO_C}" >&3
}
#
cleanup_files=''
cleanup_procs=''
#
cleanup () {
    rm -f $logfile $cleanup_files
    test -n "$cleanup_procs" && kill $cleanup_procs
}
#
test_interrupted () {
    test -z "$QUIET" && echo "${ECHO_T}INTERRUPTED" >&3
    cleanup
    exit 1
}
test_echo () {
    test -z "$QUIET" && echo "${ECHO_T}$*" >&3
}
test_cat () {
    cat $* >&3
}
#
trap test_interrupted 1 2 13 15
#
test_passed () {
    test -z "$QUIET" && echo "${ECHO_T}PASS" >&3
    cleanup
    exit 0
}
#
test_passed_continue () {
    test -z "$QUIET" && echo "${ECHO_T}PASS" >&3
    return 0
}
#
test_failed () {
    echo "${ECHO_T}FAIL" >&3
    test -n "$*" && echo "$*" >&3
    exit 1
}
#
