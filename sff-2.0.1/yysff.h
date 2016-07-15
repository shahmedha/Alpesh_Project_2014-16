// -*-C++-*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Author: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 1998-2003 University of Colorado
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
#ifndef _YYSFF_H
#define _YYSFF_H

#include "config.h"

enum sff_output_level_t {
     SFF_SILENT = 0,
     SFF_MATCH_COUNT = 1,
     SFF_VERBOSE = 2,
};
      
extern int	sff_scanner_open(const char * fname);
extern int	sff_scanner_close();
extern int	sff_scanner_is_interactive();
extern void	sff_parser_prompt();
extern int	sff_parser_stats_only();
extern int	sff_parser_run(const char * fname = 0);
extern void	sff_parser_output_off();
extern void	sff_parser_output_on();
extern void	sff_parser_output_level(sff_output_level_t l);
extern void	sff_parser_clear_statistics();
extern void	sff_parser_print_statistics(const char * frmt = 0);
extern void	sff_parser_set_statistics_format(const char * frmt);
extern void	sff_parser_set_ifconfig_ranges(const char * ranges);
extern int	sff_parser_open_output(const char * filename = 0);
extern void	sff_parser_close_output();
extern void	sff_parser_timer_start();
extern void	sff_parser_timer_stop();
#ifdef HAVE_CUDD
extern void	sff_parser_use_bddbtable();
extern void	sff_parser_use_zddbtable();
#endif
extern void	sff_parser_use_btable();
extern void	sff_parser_use_btrietable();
extern void	sff_parser_use_sorted_btable();
extern void	sff_parser_use_bxtable();
extern void	sff_parser_use_bctable();
extern void	sff_parser_use_bvtable();
extern void	sff_parser_use_fwdtable();
extern void	sff_parser_use_ttable();
extern void	sff_parser_use_tagstable();
extern void	sff_parser_shutdown();
#endif
