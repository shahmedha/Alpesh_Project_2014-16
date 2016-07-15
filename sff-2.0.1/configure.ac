dnl
dnl  This file is part of Siena, a wide-area event notification system.
dnl  See http://www.inf.usi.ch/carzaniga/siena/
dnl
dnl  Author: Antonio Carzaniga
dnl  See the file AUTHORS for full details. 
dnl
dnl  Copyright (C) 2001-2004 University of Colorado
dnl  Copyright (C) 2005 Antonio Carzaniga
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
AC_INIT(sff, 2.0.1, [Antonio Carzaniga (firstname.lastname@usi.ch)])
AC_CONFIG_AUX_DIR([config])
AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE
AM_SILENT_RULES([yes])
AM_CONFIG_HEADER(config.h:config.h.in)
AC_PREFIX_DEFAULT(/opt)
#
# libtool versioning.  Follow this rule:
# 
# if (interface changed) {
#    LT_CURRENT := LT_CURRENT + 1
#    LT_REVISION := 0
#    if (backward compatible) {
#       LT_AGE := LT_AGE + 1
#    } else {
#       LT_AGE := 0
#    }
# } else {  /* no interface changes */
#    LT_REVISION := LT_REVISION + 1
# }
#
LT_RELEASE=2
LT_CURRENT=0
LT_REVISION=1
LT_AGE=0
#
# Caveat: don't try to maintain the package version in sync with the
# ======  library version at all costs.  This is a common mistake.
#
# _LDFLAGS =  -version-info $(LT_CURRENT):$(LT_REVISION):$(LT_AGE) \
#             -release $(LT_RELEASE)
#
# LT_CURRENT is the "major" library version.  
#	Change LT_CURRENT whenever you break interface compatibility
#
# LT_REVISION is the revision number (i.e., "minor" number)
#	Change LT_CURRENT whenever you change the implementation, but
#	reset LT_VERSION=0 whenever you change LT_CURRENT
#
#
# LT_AGE indicates a compatibility range, referring to LT_CURRENT
#	e.g., 5:34:2 means also compatible with 3:*:* and 4:*:*
#
# LT_RELEASE is the label associated with the file name.  It makes sense 
#	to assign the "major" package version number to LT_RELEASE: 
#
AC_SUBST(LT_RELEASE)
AC_SUBST(LT_CURRENT)
AC_SUBST(LT_REVISION)
AC_SUBST(LT_AGE)
#
# configuration options
#
AC_OPT_PROFILING
AC_OPT_ASSERTIONS
AC_OPT_STD_BITSET
AC_OPT_DEBUGGING
#
# Checks for programs.
AC_PROG_MAKE_SET
AC_PROG_INSTALL
#
# The libtool macro AC_PROG_LIBTOOL checks for the C preprocessor.
# Configure gets confused if we try to check for a C preprocessor
# without first checking for the C compiler
# (see http://sources.redhat.com/ml/autoconf/2001-07/msg00036.html),
# so we invoke AC_PROG_CC explicitly.
AC_PROG_CC
AC_PROG_CXX
AC_LANG(C++)
#
AC_PROG_LIBTOOL
AC_PROG_YACC
AM_PROG_LEX
#
# programs
#
AC_PATH_PROG(SH, sh, /bin/sh)
AC_PATH_PROG(FALSE, false, /usr/bin/false)
AC_PATH_PROG(TRUE, true, /usr/bin/true)
AC_PATH_PROG(RM, rm, $FALSE)
AC_PATH_PROG(MV, mv, $FALSE)
AC_PATH_PROG(CP, cp, $FALSE)
AC_PATH_PROG(TAR, tar, $FALSE)
AC_PATH_PROG(GZIP, gzip, $FALSE)
#
# library, header and othere system-specific stuff
#
AC_HEADER_TIME
#
# check for doxygen
#
AC_PATH_PROG(DOXYGEN, doxygen, false, $PATH)
AM_CONDITIONAL(HAVE_DOXYGEN, test "x$DOXYGEN" != xfalse) 
HAVE_DOT=NO
if test "x$DOXYGEN" != xfalse; then
  AC_PATH_PROG(DOT, dot, false, $PATH)
  if test "x$DOT" != xfalse; then
     HAVE_DOT=YES
  fi
fi
AC_SUBST(HAVE_DOT)
#
# Boost unit testing framework
#
AX_BOOST_BASE([1.40], [
  AX_BOOST_UNIT_TEST_FRAMEWORK
  if test "x$ax_cv_boost_unit_test_framework" = "xyes"; then
      USE_BOOST_UTF=yes
  else
      USE_BOOST_UTF=no
  fi],  [USE_BOOST_UTF=no])
if test "x$USE_BOOST_UTF" != "xyes"; then
   AC_MSG_WARN([Without the Boost Unit Test Framework you will not be able to run the complete set of tests that come with SFF.])
fi
AM_CONDITIONAL([USE_BOOST_UNIT_TEST_FRAMEWORK], [test "x$USE_BOOST_UTF" = "xyes"])
#
# BDD option and CUDD libraries
#
AC_ARG_WITH(cudd-dir,
  AC_HELP_STRING([--with-cudd-dir=PATH], 
		 [Path to CUDD installation directory.]), [
      LDFLAGS="-L$withval/cudd -L$withval/mtr -L$withval/st -L$withval/util -L$withval/epd $LDFLAGS"
      CPPFLAGS="-I$withval/include $CPPFLAGS"])
#
AC_OPT_TIMERS
dnl AM_CONDITIONAL(WITH_TIMERS, test -n "$WITH_TIMERS") 
#
AC_ARG_ENABLE(bdd,
   AC_HELP_STRING([--enable-bdd],[Enable BDD-based implementation for Bloom filter matching (default=no)]), [
      AC_CHECK_HEADER([cudd.h], [], [
         AC_MSG_ERROR([CUDD include files were not found. Please pass CPPFLAGS=-I... or --with-cudd-dir=... to the configure script to point to the CUDD include directory.])])
      AC_CHECK_LIB([cudd], [Cudd_Init] , [ 
         AC_DEFINE([HAVE_CUDD], 1, [includes BDD options for Bloom filter matching implementation])
	 LIBS="-lcudd -lmtr -lst -lutil -lepd $LIBS"
        ], [AC_MSG_ERROR([CUDD libraries were not found. Please pass LDFLAGS=-L... or --with-cudd-dir=... to the configure script to point to the CUDD library directory.])], [-lmtr -lst -lutil -lepd])])
#
# allow the user to choose the size of the Bloom filters used in all
# the B* implementations.
#
AC_MSG_CHECKING(Bloom filters: size (m))
ac_bloom_filter_size=192
AC_ARG_WITH(bloom-filter-size,
  AC_HELP_STRING([--with-bloom-filter-size=SIZE],
     [Size of the Bloom filters used in BTable, BCTable, and BXTable (default 192 bits).]), [ 
     AC_MSG_RESULT($withval)
     ac_bloom_filter_size="$withval"], [ 
       AC_MSG_RESULT([$ac_bloom_filter_size (default)])])
#
# allow the user to choose the number of hash functions in our Bloom filters
# in all the B* implementations.
#
AC_MSG_CHECKING(Bloom filters: number of hash functions (k))
ac_bloom_filter_k=10
AC_ARG_WITH(bloom-filter-k,
  AC_HELP_STRING([--with-bloom-filter-k=N],
     [Number of hash functions for the Bloom filters used in BTable and its variances (default 10).]), [ 
     AC_MSG_RESULT($withval)
     ac_bloom_filter_k="$withval"], [ 
       AC_MSG_RESULT([$ac_bloom_filter_k (default)])])
#
# allow the user to select an implementation of counters that is
# faster but not reentrant.
#
AC_ARG_WITH(non-reentrant-counters,
  AC_HELP_STRING([--with-non-reentrant-counters],
     [Selects an implementation of the counting algorithm that is faster 
      but not reentrant.]), [ 
     AC_MSG_RESULT([Using fast, non-reentrant counting algorithm.])
     AC_DEFINE([WITH_STATIC_COUNTERS], 1, [uses static, faster but non-reentrant counting algorithm.])])
#
# Attributes index: TST vs. PATRICIA trie.
#
AC_ARG_WITH(attributes-index,
  AC_HELP_STRING([--with-attributes-index=Algo], 
		 [Algorithm to implement the main attributes index in an attribute-based forwarding table.  Algo can be "tst" or "patricia".  Default is "patricia".]), [
    case "$withval" in
        tst | TST )
	   AC_MSG_RESULT([Using TST for attribute index.])
	   AC_DEFINE([WITH_A_INDEX_USING_TST], 1, [uses a TST to implement the attributes index.]) 
	   ;;
	* )
	   AC_MSG_RESULT([Using PATRICIA trie for attribute index.])
           ;;
    esac
  ])
#
docdir='${prefix}/doc'
AC_ARG_WITH(docdir, 
   AC_HELP_STRING([--with-docdir=dir], [Documentation directory.]), [ 
     docdir="$withval"])
AC_SUBST(docdir)
#
AC_CHECK_BUILTIN_POPCOUNT(AC_DEFINE([HAVE_BUILTIN_POPCOUNT], 1, [uses the compiler's intrinsic __builtin_popcountl.])) 
#
AC_DEFINE_UNQUOTED([CONFIG_BLOOM_FILTER_SIZE], $ac_bloom_filter_size, [size of the Bloom filters (bits) used in the BTable, BCTable, and BXTable algorithms.])
AC_DEFINE_UNQUOTED([CONFIG_BLOOM_FILTER_K], $ac_bloom_filter_k, [number of hash functions used in the Bloom filters used in the BTable, BCTable, and BXTable algorithms.])
#
AC_CONFIG_FILES([Makefile 
	   tests/Makefile
	   doc/Makefile
	   doc/header.html])
#
AC_OUTPUT
