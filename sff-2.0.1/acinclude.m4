dnl
dnl  This file is part of Siena, a wide-area event notification system.
dnl  See http://www.inf.usi.ch/carzaniga/siena/
dnl
dnl  Author: Antonio Carzaniga
dnl  See the file AUTHORS for full details. 
dnl
dnl  Copyright (C) 2001-2003 University of Colorado
dnl
dnl  Siena is free software: you can redistribute it and/or modify
dnl  it under the terms of the GNU General Public License as published by
dnl  the Free Software Foundation, either version 3 of the License, or
dnl  (at your option) any later version.
dnl  
dnl  Siena is distributed in the hope that it will be useful,
dnl  but WITHOUT ANY WARRANTY; without even the implied warranty of
dnl  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
dnl  GNU General Public License for more details.
dnl  
dnl  You should have received a copy of the GNU General Public License
dnl  along with Siena.  If not, see <http://www.gnu.org/licenses/>.
dnl
dnl AC_DISABLE_COMPILER_OPTIMIZATION
dnl
AC_DEFUN([AC_DISABLE_COMPILER_OPTIMIZATION], [[
  CFLAGS=`echo "$CFLAGS" | sed "s/-O[^ ]*/-O0/g"`
  CXXFLAGS=`echo "$CXXFLAGS" | sed "s/-O[^ ]*/-O0/g"`
]])
dnl
dnl AC_OPT_PROFILING
dnl
AC_DEFUN([AC_OPT_PROFILING], [
AC_ARG_ENABLE(profiling, 
  AC_HELP_STRING([--enable-profiling],
	[include profiling information. Values are "yes", "coverage" and "no" (default is "no")]),
dnl this is to optionally compile with profiling
dnl I don't know too much about this, but it looks like
dnl -pg only works with static libraries, so I'm going to 
dnl disable shared libraries here.
  [ case "$enableval" in
        coverage )
	    CFLAGS_prof='-pg -fprofile-arcs -ftest-coverage'
	    CXXFLAGS_prof='-pg -fprofile-arcs -ftest-coverage'
	    LDFLAGS_prof='-pg'
	    LIBS_prof='-lgcov'
	    AC_MSG_RESULT([Enabling profiling and coverage information])
	    AC_DISABLE_COMPILER_OPTIMIZATION
	    ;;
        * )
	    CFLAGS_prof='-pg'
	    CXXFLAGS_prof='-pg'
	    LDFLAGS_prof='-pg'
	    LIBS_prof=''
	    AC_MSG_RESULT([Enabling profiling information])
	    ;;
    esac
    AC_DEFINE(WITH_PROFILING, 1, [Using gprof, so do not mess with SIGPROF])
    AC_DISABLE_SHARED ], 
  [ CFLAGS_prof=''
    CXXFLAGS_prof=''
    LDFLAGS_prof=''
    LIBS_prof=''
    AC_ENABLE_SHARED ])
AC_SUBST(CFLAGS_prof)
AC_SUBST(CXXFLAGS_prof)
AC_SUBST(LDFLAGS_prof)
AC_SUBST(LIBS_prof)
])
dnl
dnl AC_OPT_ASSERTIONS
dnl
AC_DEFUN([AC_OPT_ASSERTIONS], [
AC_ARG_ENABLE(assertions, 
  AC_HELP_STRING([--enable-assertions],
	[enable evaluation of assertions (default is NO)]), ,
  [ AC_DEFINE(NDEBUG, 1, [Disables assertions and other debugging code])])
])
dnl
dnl AC_OPT_DEBUGGING
dnl
AC_DEFUN([AC_OPT_DEBUGGING], [
AC_ARG_ENABLE(debugging, 
  AC_HELP_STRING([--enable-debugging],
	[enable debug messages of assertions (default is NO)]), 
  [ AC_DEFINE(DEBUG_OUTPUT, 1, [Enables debugging output])])
])
dnl
dnl AC_OPT_STD_BITSET
dnl
AC_DEFUN([AC_OPT_STD_BITSET], [
ac_std_bitset=no
AC_ARG_ENABLE(std_bitset, 
  AC_HELP_STRING([--enable-std-bitset],
	[enable the use of the C++ standard <bitset> (default is NO)]),
  [ ac_std_bitset=yes ])
if test "$ac_std_bitset" = yes; then
  AC_DEFINE(USE_STD_BITSET, 1, [Enables the use of C++ standard bitset])
fi
])
dnl
dnl AC_CHECK_RDTSC([action-if-available [, action-if-not-available])
dnl
AC_DEFUN([AC_CHECK_RDTSC], [
AC_CACHE_CHECK([for the rdtsc instruction], [ac_cv_rdtsc], [
AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
long long int rdtsc() {
    long long int x;
    __asm__ volatile ("rdtsc" : "=A" (x));
    return x;
}
]])], [ ac_cv_rdtsc=yes ], [ ac_cv_rdtsc=no ])])
case "$ac_cv_rdtsc" in
    yes)
	ifelse([$1], , :, [$1])
	;;
    *)
	ifelse([$2], , :, [$2])
	;;
esac
])
dnl
dnl AC_CHECK_RDTSCP([action-if-available [, action-if-not-available])
dnl
AC_DEFUN([AC_CHECK_RDTSCP], [
AC_CACHE_CHECK([for the rdtscp instruction], [ac_cv_rdtscp], [
AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
long long int rdtscp() {
    long long int x;
    __asm__ volatile ("rdtscp" : "=A" (x));
    return x;
}
]])], [ ac_cv_rdtscp=yes ], [ ac_cv_rdtscp=no ])])
case "$ac_cv_rdtscp" in
    yes)
	ifelse([$1], , :, [$1])
	;;
    *)
	ifelse([$2], , :, [$2])
	;;
esac
])
dnl
dnl AC_CHECK_MONOTONIC_RAW([action-if-available [, action-if-not-available])
dnl
AC_DEFUN([AC_CHECK_MONOTONIC_RAW], [
AC_CACHE_CHECK([for CLOCK_MONOTONIC_RAW], [ac_cv_monotonic_raw], [
AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
#include <time.h>
void f() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW,&t);
}
]])], [ ac_cv_monotonic_raw=yes ], [ ac_cv_monotonic_raw=no ])])
case "$ac_cv_monotonic_raw" in
    yes)
	ifelse([$1], , :, [$1])
	;;
    *)
	ifelse([$2], , :, [$2])
	;;
esac
])
dnl
dnl AC_OPT_TIMERS
dnl
AC_DEFUN([AC_OPT_TIMERS], [
AC_ARG_ENABLE(timers,
   AC_HELP_STRING([--enable-timers],
      [Enable performance timers. Values are "yes" or "chrono", "process", "monotonic", "monotonic_raw" or "raw", "rdtsc", and "no" (default=no)]), [
      must_test_gettime=no
      case "$enableval" in
        '' | yes | chrono )
	    AX_CXX_COMPILE_STDCXX_11(,mandatory)
	    AC_DEFINE([WITH_TIMERS], [], [libsff maintains per-module performance timers])
	    AC_DEFINE([WITH_CXX_CHRONO_TIMERS], [], [Using C++-11 std::chrono feature.])
	    ;;
	process | monotonic | monotonic_raw | raw )
	    case "$enableval" in
	    	process )
 		    AC_DEFINE([GETTIME_CLOCK_ID], [PER_PROCESS], [Per-process timer.])
		    ;;
		monotonic )
		    AC_DEFINE([GETTIME_CLOCK_ID], [MONOTONIC], [Monotonic timer.])
		    ;;
		monotonic_raw | raw )
		    AC_CHECK_MONOTONIC_RAW([
 	            AC_DEFINE([GETTIME_CLOCK_ID], [MONOTONIC_RAW], [Monotonic raw hardware timer.])
 		    ], [
		    AC_MSG_WARN([CLOCK_MONOTONIC_RAW unavailable, using clock_gettime with CLOCK_MONOTONIC.])
		    AC_DEFINE([GETTIME_CLOCK_ID], [MONOTONIC], [Monotonic timer.])
		    ])
		    ;;
	    esac
	    AC_CHECK_FUNC(clock_gettime,[
		AC_DEFINE([WITH_TIMERS], [], [libsff maintains per-module performance timers])
		], [
		    AC_CHECK_LIB(rt,clock_gettime,[
			AC_DEFINE([WITH_TIMERS], [], [libsff maintains per-module performance timers])
		    ], [
			AC_MSG_WARN([Function clock_gettime is not available.])
			AC_MSG_NOTICE([Performance timers are disabled])
		    ])
		])
	    ;;	
	rdtsc )
	    AC_CHECK_RDTSC([
	    AC_DEFINE([HAVE_RDTSC], [], [Intel rdtsc instruction])
	    AC_DEFINE([WITH_RDTSC_TIMERS], [], [implementing timers with RDTSC])
	    ], [
	    AC_MSG_WARN([RDTSC unavailable.])
	    AC_MSG_NOTICE([Performance timers are disabled])
	    ])
	    ;;
	* )
	    AC_MSG_WARN([Unknown performance timer type: "$enableval"])
	    AC_MSG_NOTICE([Performance timers are disabled])
	    ;;
	esac
  ])
])
dnl
dnl AC_CHECK_BUILTIN_POPCOUNT([action-if-available [, action-if-not-available])
dnl
AC_DEFUN([AC_CHECK_BUILTIN_POPCOUNT], [
AC_CACHE_CHECK([for the _builtin_ popcount function], [ac_cv_builtin_popcount], [
AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
unsigned int popcount(unsigned long x) {
    return __builtin_popcountl(x);
}
]])], [ ac_cv_builtin_popcount=yes ], [ ac_cv_builtin_popcount=no ])])
case "$ac_cv_builtin_popcount" in
    yes)
	ifelse([$1], , :, [$1])
	;;
    *)
	ifelse([$2], , :, [$2])
	;;
esac
])
