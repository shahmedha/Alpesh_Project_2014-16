# -*- Makefile -*-
#
#  This file is part of Siena, a wide-area event notification system.
#  See http://www.inf.usi.ch/carzaniga/siena/
#
#  Author: Antonio Carzaniga
#  See the file AUTHORS for full details. 
#
#  Copyright (C) 2001-2005 University of Colorado
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

# the default value for docdir is ${prefix}/doc.  docdir can also be
# set with the --with-docdir=dir configure option.  The default value
# ${prefix}/doc is most appropriate for an installation within a
# single (prefix) directory.  E.g., with --prefix=/opt/sff-1.8.6.  If
# you are installing sff in a common directory (e.g., --prefix=/usr)
# you might want to initialize docdir to something like this:
# docdir='${prefix}/share/doc/$(PACKAGE)-$(VERSION)'.  In any case, be
# careful not to leave any extra '/' at the end, as that tends to
# confuse install-sh
#
docdir=@docdir@

# we add all the "maintainer" sources plus the packaged documentation
# (dockpackage) as EXTRA_DIST.
#
EXTRA_DIST = $(DOCPACKAGE) fdl.txt doxygen.conf antodoc.css	\
  header.html.in footer.html driver-attributes.input		\
  driver-tags.input message_iteration.cc forwarding_messages.cc	\
  maindoc.h

# DOX_EXTRA lists the dependencies that are not directly used as input
# files to doxygen.
#
DOX_EXTRA = doxygen.conf antodoc.css header.html footer.html		\
  driver-attributes.input driver-tags.input message_iteration.cc	\
  forwarding_messages.cc

# Input files for doxygen.
#
DOX_INPUT = $(srcdir)/maindoc.h $(top_srcdir)/siena/forwarding.h	\
  $(top_srcdir)/siena/attributes.h $(top_srcdir)/siena/fwdtable.h	\
  $(top_srcdir)/siena/btable.h $(top_srcdir)/siena/bctable.h		\
  $(top_srcdir)/siena/bvtable.h $(top_srcdir)/siena/bxtable.h		\
  $(top_srcdir)/siena/bddbtable.h $(top_srcdir)/siena/btrietable.h	\
  $(top_srcdir)/siena/tags.h $(top_srcdir)/siena/tagstable.h		\
  $(top_srcdir)/siena/ttable.h

DOCPACKAGE = docpack.tar.gz 

# unfortunately, Automake does not offer good support for HTML and
# doxygen.  Worse than that, the "html" target is interpreted by
# Automake with its own rules, and is therefore unusable.  This means
# that we must stay away from the HTML_OUPTUT macro and the html
# target.  This is why I define HTMLDIR=HTML, instead of HTMLDIR=html
# or (better) HTML_OUTPUT=html.
#
HTMLDIR=HTML
#
all: $(HTMLDIR)
#
if HAVE_DOXYGEN
# I have decided that the best way to parameterize the doxygen
# configuration is through environment variables.  So, we explicitly
# set variables in DOX_ENV, which we then use when we call doxygen.
# These env variables are then referenced by doxygen.conf
DOX_ENV = top_srcdir='$(top_srcdir)'		\
	srcdir='$(srcdir)'			\
	HAVE_DOT='$(HAVE_DOT)'			\
	HTML_OUTPUT='$(HTMLDIR)'		\
	DOX_INPUT='$(DOX_INPUT)' 

$(HTMLDIR): $(DOX_INPUT) $(DOX_EXTRA) $(srcdir)/fdl.txt
	$(RM) -rf $@
	$(mkinstalldirs) $@
	$(DOX_ENV) $(DOXYGEN) $(srcdir)/doxygen.conf
	$(CP) $(srcdir)/fdl.txt $@/fdl.txt

$(DOCPACKAGE): $(HTMLDIR)
	$(TAR) cf - $(HTMLDIR) | $(GZIP) > $@

else
# a better way to do this would be:
# doc: $(DOCPACKAGE)
#	$(GZIP) -dc $< | $(TAR) xvf -
# but unfortunately some dumb versions of make, e.g., the one that
# comes with FreeBSD, don't understand the meaning of $< in this case
# 
$(HTMLDIR): $(srcdir)/$(DOCPACKAGE)
	$(RM) -rf $@
	$(GZIP) -dc $(srcdir)/$(DOCPACKAGE) | $(TAR) xvf -

endif

install-data-hook: $(HTMLDIR)
	$(mkinstalldirs) $(DESTDIR)$(docdir)
	cp -rp $(HTMLDIR) $(DESTDIR)$(docdir)

# Automake's "distcheck" is sensitive to having files left over
# after "make uninstall", so we have to clean up the install hook.
uninstall-local:
	$(RM) -rf $(DESTDIR)$(docdir)

clean-local:
	$(RM) -rf $(HTMLDIR) $(DOCPACKAGE)

MAINTAINERCLEANFILES = $(DOCPACKAGE)
