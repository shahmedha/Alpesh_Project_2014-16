/*
 *  This file is part of Siena, a wide-area event notification system.
 *  See http://www.inf.usi.ch/carzaniga/siena/
 *  
 *  Author: Antonio Carzaniga
 *  See the file AUTHORS for full details. 
 *  
 *  Copyright (C) 1998-2003 University of Colorado
 *  
 *  Siena is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  Siena is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *  
 */
%{
#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <map>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#if HAVE_UNISTD_H
# include <sys/types.h>
# include <unistd.h>
#else
# error you need unistd.h
#endif

#include <siena/forwarding.h>
#include <siena/tagstable.h>
#include <siena/ttable.h>
#include <siena/fwdtable.h>
#include <siena/btable.h>
#include <siena/btrietable.h>
#include <siena/bxtable.h>
#include <siena/bctable.h>
#include <siena/bvtable.h>
#ifdef HAVE_CUDD
#include <siena/bddbtable.h>
#endif

#include "yysff.h"
#include "yysfftypes.h"

#include "allocator.h"
#include "range_set.h"
#include "simple_attributes_types.h"
#include "simple_tags_types.h"

#include "timers.h"
#include "timing.h"

#define YYERROR_VERBOSE
#ifdef YYDEBUG
#undef YYDEBUG
#endif

bool errors = false;
int yyerror(const char *);

#ifdef WITH_TIMERS
static siena_impl::Timer sff_timer;
static siena_impl::Timer parser_timer;
#endif

static std::ofstream * sff_foutput = 0;
static std::ostream * sff_output = & std::cout;

class ICounter : public siena::MatchHandler {
public:
    ICounter() : c(0) { }
    virtual bool output(siena::InterfaceId) ;

    unsigned long long get_count() { return c; }
private:
    unsigned long long c;
};

bool ICounter::output(siena::InterfaceId x) {
    ++c;
    return false;
}

class IPrinter : public ICounter {
public:
    IPrinter(): ICounter() {};
    virtual bool output(siena::InterfaceId) ;
    void flush(std::ostream & os);

 private:
    std::set<siena::InterfaceId> iset;
};

bool IPrinter::output(siena::InterfaceId x) {
    ICounter::output(x);
    iset.insert(x);
    return false;
}

void IPrinter::flush(std::ostream & os) {
    os << "->";
    std::set<siena::InterfaceId>::const_iterator i;
    for(i = iset.begin(); i != iset.end(); ++i)
	os << ' ' << *i;
    os << std::endl;
}

bool stats_only = false; 
     
static sff_output_level_t output_level = SFF_VERBOSE; 

unsigned long i_counter;
unsigned long f_counter;
unsigned long c_counter;
unsigned long m_counter;
unsigned long a_counter;
unsigned long x_counter;
unsigned long xx_counter;
unsigned long t_counter;
unsigned long ts_counter;

static const char * StatsFormat = 0;

static range_set<int> *ifconfig_ranges = NULL;

static siena::AttributesFIB * FT = 0;
static siena::TagsFIB * TT = 0;

static bool consolidate_guard = false;

static siena_impl::batch_allocator Mem;

static siena::String new_sx_string(const string * x) {
    char * s = new (Mem) char[x->length()];
    char * t = s;
    for(string::const_iterator c = x->begin(); c != x->end(); ++c)
	 *t++ = *c;
    return siena::String(s, t);
}

static siena::String new_sx_string() {
    return siena::String(0, 0);
}

std::ostream & operator << (std::ostream & os,  const siena::Constraint & a) {

    siena::String name = a.name();
    for(const char * c = name.begin; c != name.end; ++c)
	 os << *c;

    os << ' ';
    switch(a.op()) {
    case siena::EQ: os << "="; break;
    case siena::LT: os << "<"; break;
    case siena::GT: os << ">"; break;
    case siena::SF: os << "*="; break;
    case siena::PF: os << "=*"; break;
    case siena::SS: os << "**"; break;
    case siena::ANY: os << "any"; break;
    case siena::NE: os << "!="; break;
    case siena::RE: os << "~="; break;
    }

    os << ' ';
    switch(a.type()) {
    case siena::STRING: {
	 siena::String value = a.string_value();
	 os << '"';
	 for(const char * c = value.begin; c != value.end; ++c) {
	     switch(*c) {
	     case '"': os << "\\\""; break;
	     case '\\': os << "\\\\"; break;
	     default: os << *c; break;
	     }
	 }
	 os << '"';
	 break;
    }
    case siena::INT: os << a.int_value(); break;
    case siena::DOUBLE: os << a.int_value(); break;
    case siena::BOOL: os << (a.bool_value() ? "true" : "false"); break;
    case siena::ANYTYPE: os << "any"; break;
    }
    return os;
}

std::ostream & operator << (std::ostream & os,  const siena::Predicate & p) {
    siena::Predicate::Iterator * pi;
    if ((pi = p.first())) {
	 bool first_filter = true;
	 do {
	     siena::Filter::Iterator * fi;
	     if ((fi = pi->first())) {
		 if (first_filter) {
		     first_filter = false;
		 } else {
		     os << "| ";
		 }
		 bool first_constraint = true;
		 do {
		     if (first_constraint) {
			 first_constraint = false;
		     } else {
			 os << ", ";
		     }
		     os << *fi;
		 } while (fi->next());
		 delete(fi);
		 os << std::endl;
	     }
	 } while (pi->next());
	 delete (pi);
    }
    return os;
}

static void sff_clear() {
    consolidate_guard = false;
    if (FT) {
	FT->clear();
    } else if (TT) {
	TT->clear();
    }
}

static void sff_clear_recycle() {
    consolidate_guard = false;
    if (FT) {
	FT->clear_recycle();
    } else if (TT) {
	TT->clear_recycle();
    }
}

static void sff_consolidate() {
    if (!consolidate_guard) {
	if (FT) {
	    FT->consolidate();
	    consolidate_guard = true;
	} else if (TT) {
	    TT->consolidate();
	    consolidate_guard = true;
	}
    }
}

static size_t sff_get_bytesize() {
    if (FT)
	return FT->bytesize();
    if (TT)
	return TT->bytesize();
    return 0;
}

static size_t sff_get_allocated_bytesize() {
    if (FT)
	return FT->allocated_bytesize();
    if (TT)
	return TT->allocated_bytesize();
    return 0;
}

static const char * prompt_string = 0;

static void sff_parser_complete_command() {
    prompt_string = "SFF> ";
}

void sff_parser_incomplete_command() {
    prompt_string = "...> ";
}

void sff_parser_prompt() {
    std::cout << prompt_string << std::flush;
}

static void sff_parser_first_prompt() {
    if (sff_scanner_is_interactive()) {
	*sff_output << "";
	std::cout << "Siena Fast Forwarding Module" 
	    " (" << PACKAGE_NAME << " v." << PACKAGE_VERSION << ")\n"
	    "Copyright (C) 2001-2005 University of Colorado\n"
	    "Copyright (C) 2005-2013 Antonio Carzaniga\n"
	    "This program comes with ABSOLUTELY NO WARRANTY.\n"
	    "This is free software, and you are welcome to redistribute it\n"
	    "under certain conditions. See the file COPYING for details.\n\n";

	sff_parser_complete_command();
	sff_parser_prompt();
    }
}

static const char * help_text[]  = {
    "ifconfig",
    "ifconfig <ifx_id> <predicate>;\n"
    "\tassociates interface number <ifx_id> with predicate <predicate>\n"
    "\n"
    "ifconfig <ifx_id> <tagset-list>;\n"
    "\tassociates interface number <ifx_id> with a list of tag sets <tagset-list>\n"
    ,
    "select",
    "select <message>;\n"
    "\tprocesses the given message for forwarding\n"
    "\n"
    "select <tagset>;\n"
    "\tprocesses the given tag set for forwarding\n"
    ,
    "statistics", 
    "statistics [format-string]\n"
    "\tprints the current statistics using the given format string, or the\n"
    "\tpreconfigured format string if none is given\n"
    ,
    "set",
    "set statistics_only = true\n"
    "\tonly reads and counts the input predicates and messages, without\n"
    "\tcompiling or a forwarding table.  This might be useful to count the\n"
    "\tnumber of filters, predicates, constraints, etc. in a particular\n"
    "\tfile.\n"
    "\n"
    "set preprocessing_limit = <N>;\n"
    "\tsets the number of preprocessing rounds for the FwdTable algorithm.\n"
    "\n"
    "set algorithm = <algorithm>;\n"
    "\tuses the given algorithm for the forwarding table.\n"
    "\tknown algorithms are:\n"
    "\t\tfwdtable\n"
    "\t\tbddbtable\n"
    "\t\tzddbtable\n"
    "\t\tbtable\n"
    "\t\tbtrietable\n"
    "\t\tsorted_btable\n"
    "\t\tbxtable\n"
    "\t\tbctable\n"
    "\t\tbvtable\n"
    "\t\tttable\n"
    "\t\ttagstable\n"
    ,
    "timer",
    "timer start;\n"
    "\tstarts the performance timers\n"
    "\n"
    "timer stop;\n"
    "\tstops the performance timers\n"
    ,
    "output",
    "output on;\n"
    "\tactivates the normal output of the matcher \n"
    "output off;\n"
    "\tsuppresses the normal output of the matcher\n"
    "\n"
    "output > [filename];\n"
    "\tredirects the output of the matcher to the given file, or to\n"
    "\tstandard output if a name is not given\n"
    ,
    "clear",
    "clear;\n"
    "\tclears the forwarding table.  This is necessary to reconstruct the\n"
    "\ttable (from scratch).  This operation also releases the memory allocated\n"
    "\tfor the forwwarding table.\n"
    "clear recycle;\n"
    "\tclears the forwarding table without releasing the allocated memory.\n"
    "\tThis means that the memory will be recycled for the new content of\n"
    "\tthe table.\n"
    ,
    "consolidate",
    "consolidate;\n"
    "\tconsolidates the current value of the forwarding table.  This is\n"
    "\t*necessary* before the forwarding table can be used for matching.\n"
    ,
    0
};

static void sff_parser_help(const char * command) {
    if (!command) {
	std::cout << "help available for the following commands:\n";
	for(int i = 0; help_text[i] != 0; i += 2) 
	    std::cout << help_text[i] << std::endl;
    } else {
	for(int i = 0; help_text[i] != 0; i += 2) 
	    if (strcmp(help_text[i],command) == 0) {
		std::cout << help_text[i+1];
		return;
	    }
	std::cout << "Unknown command: " << command << std::endl;
    } 
 }

%}

%pure_parser

/*
 * Keywords & other lex tokens
 */
%token <str_v> ID_v STR_v REGEX_V
%token <int_v> INT_v
%token <bool_v> BOOL_v
%token <double_v> DOUBLE_v
%token AND_op OR_op LT_op GT_op EQ_op NE_op PF_op SF_op SS_op RE_op
%token INTEGER_kw STRING_kw BOOLEAN_kw DOUBLE_kw ANY_kw
%token <nlin> IFCONFIG_kw SELECT_kw 
%token <nlin> SET_kw CONSOLIDATE_kw OUTPUT_kw STATISTICS_kw CLEAR_kw TIMER_kw HELP_kw
/*
 * some semantic types for non-terminals
 */
%type <str_v>		Id
%type <value>		Value
%type <constraint>	Constraint
%type <constraint>	AnyConstraint
%type <message>		Message
%type <predicate>	Predicate
%type <op>		Op
%type <tagset_list>	TagSetList
%type <tagset>		TagSet
%type <str_v>		Tag

%start StmtList

%%
/*
 * Grammar rules
 */
StmtList: Stmt ';' { sff_parser_complete_command(); } 
 | StmtList Stmt ';' { sff_parser_complete_command(); } 
 | error ';' { sff_parser_complete_command(); } 
;

Stmt: IfConfig 
 | Select 
 | Output
 | Clear
 | Consolidate
 | Statistics
 | Timer
 | SetParameter
 | Help
;

Help: HELP_kw
   {
       sff_parser_help(0);
   }
 | HELP_kw Id
   {
       sff_parser_help($2->c_str());
       delete($2);
   }
;

SetParameter: SET_kw Id EQ_op BOOL_v
   {
       if (*$2 == "statistics_only" || *$2 == "stats_only") {
	   stats_only = $4;
       } else {
	   std::cerr << yysfffname << ':' << $1 // implicitly nlin
		     << ": unknown Boolean parameter " << *$2 << std::endl;
       }
       delete($2);
   }
 | SET_kw Id EQ_op INT_v
   {
       if (*$2 == "preprocessing_limit") {
	   siena::FwdTable * FwdT = dynamic_cast<siena::FwdTable *>(FT);
	   if (FwdT) {
	       FwdT->set_preprocess_rounds($4); 
	   } else {
	       std::cerr << "Not using the FwdTable algorithm." << std::endl;
	   }
       } else {
	   std::cerr << yysfffname << ':' << $1 // implicitly nlin
		     << ": unknown integer parameter " << *$2 << std::endl;
       }
       delete($2);
   }
 | SET_kw Id EQ_op ID_v
   {
       if (*$2 == "algorithm") {
	   if (*$4 == "fwdtable" || *$4 == "FwdTable") {
	       sff_parser_use_fwdtable();
	   }
#ifdef HAVE_CUDD
	   else if (*$4 == "bddbtable" || *$4 == "BDDBTable") {
	       sff_parser_use_bddbtable();
	   }
	   else if (*$4 == "zddbtable" || *$4 == "ZDDBTable") {
	       sff_parser_use_zddbtable();
	   }
#endif
	   else if (*$4 == "btable" || *$4 == "BTable") {
	       sff_parser_use_btable();
	   }
	   else if (*$4 == "btrietable" || *$4 == "BTrieTable") {
	       sff_parser_use_btrietable();
	   }
	   else if (*$4 == "sorted_btable" || *$4 == "SortedBTable") {
	       sff_parser_use_sorted_btable();
	   }
	   else if (*$4 == "bxtable" || *$4 == "BXTable") {
	       sff_parser_use_bxtable();
	   }
	   else if (*$4 == "bctable" || *$4 == "BCTable") {
	       sff_parser_use_bctable();
	   }
	   else if (*$4 == "bvtable" || *$4 == "BVTable") {
	       sff_parser_use_bvtable();
	   }
	   else if (*$4 == "ttable" || *$4 == "TTable") {
	       sff_parser_use_ttable();
	   }
	   else if (*$4 == "tagstable" || *$4 == "TagsTable") {
	       sff_parser_use_tagstable();
	   }
	   else {
	   std::cerr << yysfffname << ':' << $1 // implicitly nlin
		     << ": unknown algorithm " << *$4 << std::endl;
	   }
       } else {
	   std::cerr << yysfffname << ':' << $1 // implicitly nlin
		     << ": unknown string parameter " << *$2 << std::endl;
       }
       delete($2);
       delete($4);
   }
;


Timer: TIMER_kw Id 
   {
       if (*$2 == "start") {
	   sff_parser_timer_start();
       } else if (*$2 == "stop") {
	   sff_parser_timer_stop();
       } else {
	   std::cerr << yysfffname << ':' << $1 // implicitly nlin
		     << ": unknown parameter " << *$2 << std::endl;
       }
       delete($2);
   }
;
Statistics: STATISTICS_kw 
   { 
       if (errors) 
	   std::cerr << "Warning: there were errors in the input." << std::endl;
       sff_parser_print_statistics(StatsFormat);
   }
 | STATISTICS_kw STR_v 
   { 
       if (errors) 
	   std::cerr << "Warning: there were errors in the input." << std::endl;
       sff_parser_print_statistics(($2->length() > 0) ? ($2->c_str()) : StatsFormat); 
       delete($2);
   }
;

Consolidate: CONSOLIDATE_kw { sff_consolidate(); }
;

Clear: CLEAR_kw { sff_clear(); }
  | CLEAR_kw Id 
    {
       if (*$2 == "recycle" || *$2 == "-r" || *$2 == "--recycle") {
	   sff_clear_recycle();
       } else {
	   std::cerr << yysfffname << ':' << $1 // implicitly nlin
		     << ": unknown parameter " << *$2 << std::endl;
       }
       delete($2);
    }
;

Output: OUTPUT_kw Id 
   {
       if (*$2 == "on") {
	   output_level = SFF_VERBOSE;
       } else if (*$2 == "off") {
	   output_level = SFF_SILENT; 
       } else if (*$2 == "count") {
	   output_level = SFF_MATCH_COUNT; 
       } else {
	   std::cerr << yysfffname << ':' << $1 // implicitly nlin
		     << ": unknown parameter " << *$2 << std::endl;
       }
       delete($2);
   }
 | OUTPUT_kw GT_op STR_v
   {
       sff_parser_open_output($3->c_str()); 
       delete($3);
   }
 | OUTPUT_kw GT_op '-'
   { 
       sff_parser_open_output(0); 
   }
 ;

IfConfig: IFCONFIG_kw INT_v Predicate
   {
       if (! stats_only) {
	   if (ifconfig_ranges && !(*ifconfig_ranges)[$2])
	       goto cleanup_ifat;
	    if (!FT) {
		if (TT) {
		    std::cerr << yysfffname << ':' << $1 // implicitly nlin
			      << ": tags table already active" << std::endl;
		    goto cleanup_ifat;
		} else {
		    // default FT:
		    sff_parser_use_fwdtable();
		}
	    } 
	    if (consolidate_guard) {
		std::cerr << yysfffname << ':' << $1 // implicitly nlin
			  << ": forwarding table already consolidated" << std::endl;
	    } else {
		try {
		    FT->ifconfig($2, *$3);
		} catch (siena::BadConstraint & ex) {
		    std::cerr << yysfffname << ':' << $1 // implicitly nlin
			      << ": bad constraint in predicate: " 
			      << ex.what() << std::endl;
		}
	    }
	cleanup_ifat:
	    Mem.recycle();
	    delete($3);
	}
	++i_counter;
    }
  | IFCONFIG_kw INT_v TagSetList
    {
	if (! stats_only) {
	    if (ifconfig_ranges && !(*ifconfig_ranges)[$2])
		goto cleanup_iftt;
	    if (!TT) {
		if (FT) {
		    std::cerr << yysfffname << ':' << $1 // implicitly nlin
			      << ": forwarding table already active" << std::endl;
		    goto cleanup_iftt;
		} else {
		    // default TT:
		    sff_parser_use_tagstable();
		}
	    } 
	    if (consolidate_guard) {
		std::cerr << yysfffname << ':' << $1 // implicitly nlin
			  << ": tags table already consolidated" << std::endl;
	    } 
	    try {
		TT->ifconfig($2, *$3);
	    } catch (siena::BadConstraint & ex) {
		std::cerr << yysfffname << ':' << $1 // implicitly nlin
			  << ": bad constraint in predicate: " 
			  << ex.what() << std::endl;
	    }
	cleanup_iftt:
	    Mem.recycle();
	    delete($3);
	}
	++i_counter;
    }
 ;

 Id: ID_v { $$ = $1; }
  | INTEGER_kw { $$ = new std::string("integer"); }
  | STRING_kw { $$ = new std::string("string"); }
  | BOOLEAN_kw { $$ = new std::string("boolean"); }
  | DOUBLE_kw { $$ = new std::string("double"); }
  | IFCONFIG_kw { $$ = new std::string("ifconfig"); }
  | SELECT_kw { $$ = new std::string("select"); }
  | SET_kw { $$ = new std::string("set"); }
  | CONSOLIDATE_kw { $$ = new std::string("consolidate"); }
  | OUTPUT_kw { $$ = new std::string("output"); }
  | TIMER_kw { $$ = new std::string("timer"); }
  | HELP_kw { $$ = new std::string("help"); }
  | STATISTICS_kw { $$ = new std::string("statistics"); }
  | CLEAR_kw { $$ = new std::string("clear"); }
  | BOOL_v { $$ = ($1) ? (new std::string("true")) : (new std::string("false")); }
  | STR_v { $$ = $1; }
 ;

 Predicate: Id Constraint 
    {
	if (! stats_only) {
	    $$ = new simple_predicate();
	    $$->add(new simple_filter());
	    $$->last_filter()->add(new_sx_string($1), $2);
	}
	++f_counter;
	++c_counter;
	delete($1);
    }
  | Predicate AND_op Id Constraint
    {
	if (! stats_only) {
	    $$ = $1;
	    $$->last_filter()->add(new_sx_string($3), $4);
	}
	++c_counter;
	delete($3);
    }
  | Predicate OR_op Id Constraint
    {
	if (! stats_only) {
	    $$ = $1;
	    $$->add(new simple_filter());
	    $$->last_filter()->add(new_sx_string($3), $4);
	}
	++f_counter;
	++c_counter;
	delete($3);
    }
 ;

 Constraint: AnyConstraint { if (!stats_only) { $$ = $1; } }
  | Op INT_v 
    { 
	if (!stats_only) { $$ = new simple_op_value($1, $2); } 
    }
  | Op ANY_kw
    { 
	if (!stats_only) { $$ = new simple_op_value($1); } 
    }
  | Op DOUBLE_v
    { 
	if (!stats_only) { $$ = new simple_op_value($1, $2); } 
    }
  | Op BOOL_v
    { 
	if (!stats_only) { $$ = new simple_op_value($1, $2); } 
    }
  | Op STR_v
    { 
	if (!stats_only) { $$ = new simple_op_value($1, new_sx_string($2)); }
	delete($2); 
    }
  | RE_op REGEX_V
    {
	if (!stats_only) { $$ = new simple_op_value(siena::RE, new_sx_string($2)); }
	delete($2); 
    }
 ;

 AnyConstraint: ANY_kw INTEGER_kw
    { 
	if (!stats_only) { 
	    $$ = new simple_op_value(siena::ANY, (siena::Int)0); 
	} 
    }
  | ANY_kw ANY_kw 
    { 
	if (!stats_only) { $$ = new simple_op_value(siena::ANY); } 
    }
  | ANY_kw DOUBLE_kw
    { 
	if (!stats_only) { 
	    $$ = new simple_op_value(siena::ANY, (siena::Double).0);
	} 
    }
  | ANY_kw BOOLEAN_kw
    { 
	if (!stats_only) { 
	    $$ = new simple_op_value(siena::ANY, false);
	} 
    }
  | ANY_kw STRING_kw
    { 
	if (!stats_only) { 
	    $$ = new simple_op_value(siena::ANY, new_sx_string()); 
	} 
    }
 ;

 Op: LT_op { $$ = siena::LT; }
  |  GT_op { $$ = siena::GT; }
  |  EQ_op { $$ = siena::EQ; }
  |  NE_op { $$ = siena::NE; }
  |  PF_op { $$ = siena::PF; }
  |  SF_op { $$ = siena::SF; }
  |  SS_op { $$ = siena::SS; }
 ;

 Value: INT_v { if (!stats_only) { $$ = new simple_value($1); } }
  |  DOUBLE_v { if (!stats_only) { $$ = new simple_value($1); } }
  |  BOOL_v   { if (!stats_only) { $$ = new simple_value($1); } }
  |  ANY_kw   { if (!stats_only) { $$ = new simple_value(); } }
  |  STR_v    { if (!stats_only) { $$ = new simple_value(new_sx_string($1)); }
		delete($1); } 
 ;

 TagSetList: '{' TagSet '}'
    {
	if (!stats_only) { 
	    $$ = new simple_tagset_list();
	    $$->add_tagset($2);
	}
    }
   | TagSetList opt_Comma '{' TagSet '}'
    {
	if (!stats_only) { 
	    $$ = $1;
	    $$->add_tagset($4);
	}
    }
 ;

 TagSet: Tag
    {
	if (!stats_only) { 
	    $$ = new simple_tagset();
	    $$->add_tag(*($1));
	    delete($1);
	}
	++ts_counter;
	++t_counter;
    }
   | TagSet opt_Comma Tag
    {
	if (!stats_only) { 
	    $$ = $1;
	    $$->add_tag(*($3));
	    delete($3);
	}
	++t_counter;
    }

 opt_Comma: AND_op | /* empty */
 ;

 Tag: Id { $$ = $1; }
 ;

 Select: SELECT_kw Message
    {
	if (! stats_only) {
	    if (!FT) {
		std::cerr << yysfffname << ':' << $1 // implicitly nlin
			  << ": forwarding table not active" << std::endl;
	    } else {
		if (!consolidate_guard) {
		    FT->consolidate();
		    consolidate_guard = true;
		} 
		switch (output_level) {
		case SFF_MATCH_COUNT:
		case SFF_SILENT: {
		    ICounter cc;

		    FT->match(*$2, cc);

		    if (cc.get_count() > 0) ++x_counter;
		    xx_counter += cc.get_count();
		    if (output_level == SFF_MATCH_COUNT) {
			(*sff_output) << cc.get_count() << std::endl;
		    }
		    break;
		}
		case SFF_VERBOSE: {    
		    IPrinter ccc;

		    FT->match(*$2, ccc);

		    if (ccc.get_count() > 0) ++x_counter;
		    xx_counter += ccc.get_count();
		    ccc.flush(*sff_output);
		}
		}
	    }
	    Mem.recycle();
	    delete($2);
	}
	++m_counter;
    }
   | SELECT_kw '{' TagSet '}'
    {
	if (! stats_only) {
	    if (!TT) {
		std::cerr << yysfffname << ':' << $1 // implicitly nlin
			  << ": forwarding table not active" << std::endl;
	    } else {
		if (!consolidate_guard) {
		    TT->consolidate();
		    consolidate_guard = true;
		} 
		switch (output_level) {
		case SFF_MATCH_COUNT:
		case SFF_SILENT: {
		    ICounter cc;

		    TT->match(*$3, cc);

		    if (cc.get_count() > 0) ++x_counter;
		    xx_counter += cc.get_count();
		    if (output_level == SFF_MATCH_COUNT) {
			(*sff_output) << cc.get_count() << std::endl;
		    }
		}
		    break;
		case SFF_VERBOSE: {    
		    IPrinter ccc;

		    TT->match(*$3, ccc);

		    if (ccc.get_count() > 0) ++x_counter;
		    xx_counter += ccc.get_count();
		    ccc.flush(*sff_output);
		}
		}
	    }
	    Mem.recycle();
	    delete($3);
	}
	++m_counter;
    }
 ;

 Message: Id EQ_op Value
    { 
	if (! stats_only) {
	    $$ = new simple_message();
	    $$->add(new_sx_string($1), $3);
	}
	++a_counter;
       delete($1);        
   }
  | Message Id EQ_op Value
   {
       if (! stats_only) {
	   $$ = $1;
	   $$->add(new_sx_string($2), $4);
       }
       ++a_counter;
       delete($2); 
   }
;

%%

int sff_parser_run(const char * fname) 
{
    if (sff_scanner_open(fname)) {
	std::cerr << "couldn't read " << fname << std::endl;
	return -1;
    }

    bool sync_state = ios_base::sync_with_stdio(false);

    sff_parser_first_prompt();
    sff_parser_clear_statistics();

    TIMER_PUSH(parser_timer);
    int res = yyparse();
    TIMER_POP();

    sff_output->flush();
    ios_base::sync_with_stdio(sync_state);

    sff_parser_shutdown();
    sff_scanner_close();
    return res;
}

#if 0
int sff_parser_stats_only() 
{
    stats_only = true;
    return sff_parser_run();
}
#endif

void sff_parser_output_off() 
{
    output_level = SFF_SILENT; 
}

void sff_parser_output_on() 
{
    output_level = SFF_VERBOSE; 
}

void sff_parser_output_level(sff_output_level_t l) 
{
    output_level = l;
}

void sff_parser_timer_start() 
{
#ifdef WITH_TIMERS
    sff_timer.start();
#endif
}

void sff_parser_timer_stop() 
{
#ifdef WITH_TIMERS
    sff_timer.stop();
#endif
}

void sff_clear_tables() {
    if (FT) {
	delete(FT);
	FT = 0;
    }
    if (TT) {
	delete(TT);
	TT = 0;
    }
    consolidate_guard = false;
}

void sff_parser_shutdown() 
{
    sff_parser_close_output();
    Mem.clear();
    sff_clear_tables();
}

void sff_parser_use_btable() 
{
    sff_clear_tables();
    FT = siena::BTable::create();
}

void sff_parser_use_btrietable() 
{
    sff_clear_tables();
    FT = siena::BTrieTable::create();
}

void sff_parser_use_sorted_btable() 
{
    sff_clear_tables();
    FT = siena::SortedBTable::create();
}

void sff_parser_use_bxtable() 
{
    sff_clear_tables();
    FT = siena::BXTable::create();
}

void sff_parser_use_bctable() 
{
    sff_clear_tables();
    FT = siena::BCTable::create();
}

void sff_parser_use_bvtable() 
{
    sff_clear_tables();
    FT = siena::BVTable::create();
}

#ifdef HAVE_CUDD
void sff_parser_use_bddbtable() 
{
    sff_clear_tables();
    FT = siena::BDDBTable::create();
}

void sff_parser_use_zddbtable() 
{
    sff_clear_tables();
    FT = siena::ZDDBTable::create();
}
#endif

void sff_parser_use_fwdtable() 
{
    sff_clear_tables();
    FT = siena::FwdTable::create();
}

void sff_parser_use_ttable() 
{
    sff_clear_tables();
    TT = siena::TTable::create();
}

void sff_parser_use_tagstable() 
{
    sff_clear_tables();
    TT = siena::TagsTable::create();
}

void sff_parser_set_ifconfig_ranges(const char * ranges) 
{
    ifconfig_ranges = new range_set<int> (ranges);
}

void sff_parser_set_statistics_format(const char * frmt) 
{
    StatsFormat = frmt;
}

void sff_parser_close_output() 
{
    if (sff_foutput) {
	sff_foutput->close();
	delete (sff_foutput);
	sff_foutput = 0;
	sff_output = & std::cout;
    }
}

int sff_parser_open_output(const char * filename) 
{
    sff_parser_close_output();
    if (filename) {
	sff_foutput = new std::ofstream(filename);
	if (sff_foutput->fail()) {
	    delete (sff_foutput);
	    sff_foutput = 0;
	    return -1;
	}
	sff_output = sff_foutput;
    } else {
	sff_output = & std::cout;
    }
    return 0;
}

void sff_parser_print_statistics(const char * format) 
{
    if (!format) 
	format = "i=%i f=%f c=%c n=%n a=%a w=%w W=%W m=%m M=%M s=%s\n"
#ifdef WITH_TIMERS
	    "timers (milliseconds):\n"
	    "\tsff =\t\t%Tt\n"
	    "\tparser =\t%Tp\n"
	    "\tifconfig =\t%Ti\n"
	    "\tconsolidate =\t%Tc\n"
	    "\tencoding =\t%Te\n"
	    "\tmatch =\t\t%Tm\n"
	    "\tstring match =\t%Ts\n"
	    "\tforward =\t%Tf\n"
#endif
;

    for(const char * f = format; *f != '\0'; ++f) {
	switch(*f) {
	case '%':
	    ++f;
	    switch(*f) {
	    case 'i': (*sff_output) << i_counter; break;
	    case 'f': (*sff_output) << f_counter; break;
	    case 'c': (*sff_output) << c_counter; break;
	    case 'n': (*sff_output) << m_counter; break;
	    case 'a': (*sff_output) << a_counter; break;
	    case 'm': (*sff_output) << x_counter; break;
	    case 'M': (*sff_output) << xx_counter; break;
	    case 'w': (*sff_output) << t_counter; break;
	    case 'W': (*sff_output) << ts_counter; break;
	    case 's': (*sff_output) << sff_get_bytesize(); break;
	    case 'S': (*sff_output) << sff_get_allocated_bytesize(); break;
	    case '%': (*sff_output) << '%'; break;
#ifdef WITH_TIMERS
	    case 'T': {
		++f;
		switch (*f) {
		case 't': (*sff_output) << (sff_timer.read_microseconds()/1000); break;
		case 'p': (*sff_output) << (parser_timer.read_microseconds()/1000); break;
		case 'i': (*sff_output) << (siena_impl::ifconfig_timer.read_microseconds()/1000); break;
		case 'c': (*sff_output) << (siena_impl::consolidate_timer.read_microseconds()/1000); break;
		case 'e': (*sff_output) << (siena_impl::bloom_encoding_timer.read_microseconds()/1000); break;
		case 'm': (*sff_output) << (siena_impl::match_timer.read_microseconds()/1000); break;
		case 's': (*sff_output) << (siena_impl::string_match_timer.read_microseconds()/1000); break;
		case 'f': (*sff_output) << (siena_impl::forward_timer.read_microseconds()/1000); break;
		default:
		    std::cerr << "print_statistics: bad format character: '%T" 
			      << *f << "'" << std::endl;
		    if (*f == '\0') return;
		}
		break;
	    }
#endif
	    default:
		std::cerr << "print_statistics: bad format character: '%" 
			  << *f << "'" << std::endl;
		if (*f == '\0') return;
	    }
	    break;

	case '\\': 		
	    ++f;
	    switch(*f) {
	    case 'n': (*sff_output) << '\n'; break;
	    case 't': (*sff_output) << '\t'; break;
	    case 'a': (*sff_output) << '\a'; break;
	    case 'v': (*sff_output) << '\v'; break;
	    case '\\': (*sff_output) << '\\'; break;
	    default:
		std::cerr << "print_statistics: bad escape character: '" 
			  << *f << "'" << std::endl;
		if (*f == '\0') return;
	    }
	    break;
	default:
	    (*sff_output) << *f;
	}
    }
    sff_output->flush();
}

void sff_parser_clear_statistics() 
{
    i_counter = 0;
    f_counter = 0;
    c_counter = 0;
    m_counter = 0;
    a_counter = 0;
    x_counter = 0;
    xx_counter = 0;
    t_counter = 0;
    ts_counter = 0;
#ifdef WITH_TIMERS
    sff_timer.reset();
    parser_timer.reset();
    siena_impl::ifconfig_timer.reset();
    siena_impl::consolidate_timer.reset();
    siena_impl::match_timer.reset();
    siena_impl::bloom_encoding_timer.reset();
    siena_impl::string_match_timer.reset();
#endif
}

int yyerror(const char *s) 
{
    std::cerr << yysfffname << ':' << yylineno << ": " << s << std::endl;
    errors = true;
    return -1;
}
