// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2002-2004 University of Colorado
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
#include <exception>
#include <iostream>
#include <cstring>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "yysff.h"

void print_usage(const char * progname) {
    std::cout << "Siena Fast Forwarding Module" 
	      << "(" << PACKAGE_NAME << " v." << PACKAGE_VERSION << ")\n\
Copyright (C) 2001-2005 University of Colorado\n\
Copyright (C) 2005-2012 Antonio Carzaniga\n\
This program comes with ABSOLUTELY NO WARRANTY.\n\
This is free software, and you are welcome to redistribute it\n\
under certain conditions. See the file COPYING for details.\n\
\nusage: " << progname << " -f <algo> | -p | -i <ranges> | [-o <stats-format>] | [-O <filename>] [--] [<filename>...]\n\
    -f <algo>  selects a forwarding algorithm/table.\n\
       The following algorithms are available:\n\
       Attribute-based tables:\n\
          d  uses the FwdTable algorithm (default).\n\
          b  uses the BTable algorithm.\n\
          t  uses the BTrieTable algorithm.\n\
          s  uses the SortedBTable algorithm.\n\
          X  uses the BXTable algorithm.\n\
          c  uses the BCTable algorithm.\n\
          v  uses the BVTable algorithm.\n"
#ifdef HAVE_CUDD
"          B  uses the BDDBTable algorithm.\n\
          Z  uses the ZDDBTable algorithm.\n"
#endif
"       Tag-based tables:\n\
          T  uses the TagsTable algorithm.\n\
          Tt  uses the TTable algorithm.\n\
\n\
    -q  quiet.  Suppresses matching output.\n\
    -c  prints only the total number (count) of interfaces matched by each message.\n\
    -p  prints statistics after processing each input file.\n\
    -i <ranges>  only enable the given interfaces (e.g., 1-3,7).\n\
    -O <filename>  writes results and statistics on the given output file.\n\
    -o <stats-format>  uses the given output format for statistics.\n\
       stats-format is a printf-style format string.\n\
       The following format directives are defined:\n\
          %i  interface count\n\
          %n  message count\n\
          %c  constraint count\n\
          %f  filter count\n\
          %a  attribute count\n\
          %w  tag count\n\
          %W  tagset count\n\
          %m  number of messages matching at least one interface\n\
          %M  total number of matches\n\
          %s  size of the forwarding table in bytes\n\
          %S  total memory allocated by the forwarding table in bytes\n"
#ifdef WITH_TIMERS
"          %T<x> timers in milliseconds\n\
                where <x> can be: \n\
                t=sff library functions, p=parsing,\n\
                i=ifconfig, c=consolidate, m=matching, \n\
                e=message encoding, s=string index, f=forwarding\n"
#endif
	      << std::endl;
}

int main(int argc, char *argv[]) {
    int res = 0, i = 0;
    bool print_stats = false;
    const char * stats_format = 0;
    try {
	while (++i < argc) {
	    if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
		print_usage(argv[0]);
		return 0;
	    } else if (strcmp(argv[i], "-p") == 0) {
		print_stats = true;
	    } else if (strcmp(argv[i], "-i") == 0) {
		if (++i < argc) {
		    sff_parser_set_ifconfig_ranges(argv[i]);
		} else {
		    print_usage(argv[0]);
		    return 1;
		}
	    } else if (strcmp(argv[i], "-o") == 0) {
		if (++i < argc) {
		    sff_parser_set_statistics_format(stats_format = argv[i]);
		} else {
		    print_usage(argv[0]);
		    return 1;
		}
	    } else if (strcmp(argv[i], "-O") == 0) {
		if (++i < argc) {
		    if (sff_parser_open_output(argv[i]) < 0) {
			std::cerr << "error opening output file `" 
				  << argv[i] << "'" << std::endl;
			return 1;
		    }
		} else {
		    print_usage(argv[0]);
		    return 1;
		}
	    } else if (strcmp(argv[i], "-q") == 0) {
		sff_parser_output_off();
	    } else if (strcmp(argv[i], "-c") == 0) {
		sff_parser_output_level(SFF_MATCH_COUNT);
	    } else if (strcmp(argv[i], "-f") == 0) {
		if (++i < argc) {
		    switch (*argv[i]) {
		    case 'd': sff_parser_use_fwdtable(); break;
		    case 'b': sff_parser_use_btable(); break;
		    case 't': sff_parser_use_btrietable(); break;
		    case 's': sff_parser_use_sorted_btable(); break;
		    case 'X': sff_parser_use_bxtable(); break;
		    case 'c': sff_parser_use_bctable(); break;
		    case 'v': sff_parser_use_bvtable(); break;
#ifdef HAVE_CUDD
		    case 'B': sff_parser_use_bddbtable(); break;
		    case 'Z': sff_parser_use_zddbtable(); break;
#endif
		    case 'T': 
			switch (argv[i][1]) {
			case '\0': sff_parser_use_tagstable(); break;
			case 't': sff_parser_use_ttable(); break;
			default:
			    std::cerr << "bad algorithm selector `" 
				      << *argv[i] << "'" << std::endl;
			    return 1;
			}
			break;
		    default:
			std::cerr << "bad algorithm selector `" 
				  << *argv[i] << "'" << std::endl;
			return 1;
		    }
		} else {
		    print_usage(argv[0]);
		    return 1;
		}
	    } else {
		if (strcmp(argv[i], "--") == 0)
		    ++i;
		break;
	    }
	}

	sff_parser_clear_statistics();

	if (i < argc) {
	    do {
		res = sff_parser_run(argv[i]);
	    } while (++i < argc);
	} else {
	    res = sff_parser_run();
	} 

	if (print_stats)
	    sff_parser_print_statistics(stats_format);

	sff_parser_shutdown();
	return res;
    } catch (std::exception & ex) {
	std::cerr << ex.what() << std::endl;
	return 3;
    }
}
