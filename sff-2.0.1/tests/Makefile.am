# -*- Makefile -*-
#
#  This file is part of Siena, a wide-area event notification system.
#  See http://www.inf.usi.ch/carzaniga/siena/
#
#  Author: Antonio Carzaniga
#  See the file AUTHORS for full details. 
#
#  Copyright (C) 2001-2003 University of Colorado
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
AM_CPPFLAGS = -I$(top_srcdir)
AM_CXXFLAGS = -Wall
AM_LDFLAGS = -L$(top_builddir)

SFFFLAGS = 
check_PROGRAMS = 
if USE_BOOST_UNIT_TEST_FRAMEWORK
check_PROGRAMS += test_counters_map test_pointers_set test_allocator test_bitvector test_timers
endif

test_counters_map_SOURCES = test_counters_map.cc
test_counters_map_LDADD = $(BOOST_UNIT_TEST_FRAMEWORK_LIB) -lsff

test_pointers_set_SOURCES = test_pointers_set.cc
test_pointers_set_LDADD = $(BOOST_UNIT_TEST_FRAMEWORK_LIB) -lsff

test_allocator_SOURCES = test_allocator.cc ../simple_tags_types.h ../simple_tags_types.cc
test_allocator_LDADD = $(BOOST_UNIT_TEST_FRAMEWORK_LIB) -lsff

test_bitvector_SOURCES = test_bitvector.cc
test_bitvector_LDADD = $(BOOST_UNIT_TEST_FRAMEWORK_LIB) -lsff

test_timers_SOURCES = test_timers.cc
test_timers_LDADD = $(BOOST_UNIT_TEST_FRAMEWORK_LIB) -lsff

TESTDATA = any1.good any1.inp			\
	any2.inp any2.good			\
	any3.inp any3.good			\
	any4.inp any4.good			\
	complete.inp complete.good		\
	clear1.inp clear1.good			\
	clear2.inp clear2.good			\
	eq1.inp eq1.good			\
	eq2.inp eq2.good			\
	eq3.inp eq3.good			\
	eq4.inp eq4.good			\
	eq5.inp eq5.good			\
	gt1.inp gt1.good			\
	gt2.inp gt2.good			\
	gt3.inp gt3.good			\
	gt4.inp gt4.good			\
	lt1.inp lt1.good			\
	lt2.inp lt2.good			\
	lt3.inp lt3.good			\
	lt4.inp lt4.good			\
	bool.inp bool.good			\
	misc1.inp misc1.good			\
	misc2.inp misc2.good			\
	names.inp names.good			\
	ne1.inp ne1.good			\
	ne2.inp ne2.good			\
	selectivity1.inp selectivity1.good 	\
	selectivity2.inp selectivity2.good 	\
	strings1.inp strings1.good		\
	strings2.inp strings2.good		\
	strings3.inp strings3.good		\
	strings4.inp strings4.good		\
	strings5.inp strings5.good		\
	strings6.inp strings6.good		\
	sff1.inp sff1.good			\
	sff2.inp sff2.good			\
	tags_sanitized.inp tags_sanitized.good 

TESTSCRIPTS = names.sh \
	any.sh eq.sh ne.sh gt.sh lt.sh selectivity.sh \
	strings.sh bool.sh misc.sh complete.sh bloom_encoding.sh \
	clear.sh sff.sh tags.sh

UNIT_TESTS = 
if USE_BOOST_UNIT_TEST_FRAMEWORK
UNIT_TESTS += test_counters_map test_pointers_set test_allocator test_bitvector test_timers
endif

INTEGRATION_TESTS = $(TESTSCRIPTS)

TESTS = $(UNIT_TESTS) $(INTEGRATION_TESTS)

TESTS_ENVIRONMENT = QUIET=1 SFF='$(top_builddir)/sff $(SFFFLAGS)' \
	BOOST_TEST_LOG_LEVEL=test_suite
EXTRA_DIST = config.sh $(TESTSCRIPTS) $(TESTDATA)

test: run-unit-tests run-integration-tests 

run-unit-tests: $(UNIT_TESTS)
	@list='$(UNIT_TESTS)'; \
	echo "Running unit tests..."; \
        if test -n "$$list"; then \
          for tst in $$list; do \
	    $(TESTS_ENVIRONMENT) ./$$tst || exit 1; \
	  done; \
	fi

run-integration-tests: $(INTEGRATION_TESTS)
	@MAKE=$(MAKE); export MAKE; \
	srcdir=$(srcdir); export srcdir; \
        list='$(INTEGRATION_TESTS)'; \
	echo "Running integration tests..."; \
        if test -n "$$list"; then \
          for tst in $$list; do \
	    $(srcdir)/$$tst || exit 1; \
	  done; \
	fi

clean-local:
	$(RM) -rf bv pat test.out test.log test.in
