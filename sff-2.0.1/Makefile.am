# -*- Makefile -*-
#
#  This file is part of Siena, a wide-area event notification system.
#  See http://www.inf.usi.ch/carzaniga/siena/
#
#  Author: Antonio Carzaniga
#  See the file AUTHORS for full details. 
#
#  Copyright (C) 2001-2005 University of Colorado
#  Copyright (C) 2005 Antonio Carzaniga
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
SUBDIRS = . tests doc
#
SUFFIXES = .cc .h .yy .ll
#
ACLOCAL_AMFLAGS = -I m4
#
lib_LTLIBRARIES = libsff.la
libsff_la_SOURCES = \
	attributes.cc allocator.cc allocator.h \
	timing.cc timing.h timers.h timers.cc \
	fwd_table.h fwd_table.cc \
	a_index.h a_index.cc constraint_index.h counters_map.h counters_map.cc \
	pointers_set.h pointers_set.cc \
	string_index.h string_index.cc bool_index.h bool_index.cc \
	bitvector.h bitvector.icc bitvector.cc \
	v_index.h \
	hash.h hash.cc \
	b_table.h b_table.cc attributes_encoding.h \
	bloom_filter.h b_predicate.h \
	btrietable.cc \
	bc_table.cc \
	bx_table.cc \
	bv_table.cc \
	bdd_table.cc zdd_table.cc \
	t_table.cc

libsffincludedir = $(includedir)/siena
libsffinclude_HEADERS = siena/forwarding.h siena/attributes.h siena/tags.h \
	siena/fwdtable.h siena/btable.h siena/bxtable.h siena/bddbtable.h \
	siena/bctable.h siena/bvtable.h siena/btrietable.h \
	siena/tagstable.h siena/ttable.h

nobase_include_HEADERS = siena/forwarding.h siena/attributes.h \
	siena/fwdtable.h siena/btable.h siena/bxtable.h siena/bddbtable.h \
	siena/bctable.h siena/bvtable.h siena/btrietable.h \
	siena/tagstable.h siena/ttable.h

AM_YFLAGS = -d

AM_CFLAGS = -Wall $(CFLAGS_prof) $(BOOST_CPPFLAGS)
AM_CXXFLAGS = -Wall $(CXXFLAGS_prof) $(BOOST_CPPFLAGS)
AM_LDFLAGS = $(LDFLAGS_prof) $(BOOST_LDFLAGS)

libsff_la_LDFLAGS = \
 -version-info $(LT_CURRENT):$(LT_REVISION):$(LT_AGE) \
 -release $(LT_RELEASE)

CLEANFILES = sff_parser.hh sff_parser.cc sff_scanner.cc
# 
# I explicitly define the dependency sff_scanner.o: sff_parser.hh. An
# alternative is to define BUILT_SOURCES as follows:
#
#   BUILT_SOURCES = sff_parser.hh sff_parser.cc sff_scanner.cc
#
sff_scanner.$(OBJEXT): sff_parser.hh

bin_PROGRAMS = sff
sff_SOURCES = sff_parser.yy sff_scanner.ll sff.cc \
	yysff.h yysfftypes.h \
	simple_attributes_types.h simple_attributes_types.cc \
	simple_tags_types.h simple_tags_types.cc \
	range_set.h 
sff_LDADD = $(LIBS_prof) libsff.la

.PHONY: release snapshot doc test rpm

release:
	rm -rf .deps */.deps
	$(MAKE) distcheck

snapshot:
	$(MAKE) dist distdir=$(PACKAGE)-`date +%Y-%m-%d`

test:
	cd tests && $(MAKE) test

doc:
	cd doc && $(MAKE) all

#
# cleaning up...
#
DISTCLEANFILES = config/stamp-h1 so_locations
