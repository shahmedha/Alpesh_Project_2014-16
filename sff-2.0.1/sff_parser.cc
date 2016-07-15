/* A Bison parser, made by GNU Bison 3.0.2.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2013 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "3.0.2"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 1

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* Copy the first part of user declarations.  */
#line 21 "sff_parser.yy" /* yacc.c:339  */

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


#line 446 "sff_parser.cc" /* yacc.c:339  */

# ifndef YY_NULLPTR
#  if defined __cplusplus && 201103L <= __cplusplus
#   define YY_NULLPTR nullptr
#  else
#   define YY_NULLPTR 0
#  endif
# endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* In a future release of Bison, this section will be replaced
   by #include "y.tab.h".  */
#ifndef YY_YY_SFF_PARSER_HH_INCLUDED
# define YY_YY_SFF_PARSER_HH_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token type.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    ID_v = 258,
    STR_v = 259,
    REGEX_V = 260,
    INT_v = 261,
    BOOL_v = 262,
    DOUBLE_v = 263,
    AND_op = 264,
    OR_op = 265,
    LT_op = 266,
    GT_op = 267,
    EQ_op = 268,
    NE_op = 269,
    PF_op = 270,
    SF_op = 271,
    SS_op = 272,
    RE_op = 273,
    INTEGER_kw = 274,
    STRING_kw = 275,
    BOOLEAN_kw = 276,
    DOUBLE_kw = 277,
    ANY_kw = 278,
    IFCONFIG_kw = 279,
    SELECT_kw = 280,
    SET_kw = 281,
    CONSOLIDATE_kw = 282,
    OUTPUT_kw = 283,
    STATISTICS_kw = 284,
    CLEAR_kw = 285,
    TIMER_kw = 286,
    HELP_kw = 287
  };
#endif
/* Tokens.  */
#define ID_v 258
#define STR_v 259
#define REGEX_V 260
#define INT_v 261
#define BOOL_v 262
#define DOUBLE_v 263
#define AND_op 264
#define OR_op 265
#define LT_op 266
#define GT_op 267
#define EQ_op 268
#define NE_op 269
#define PF_op 270
#define SF_op 271
#define SS_op 272
#define RE_op 273
#define INTEGER_kw 274
#define STRING_kw 275
#define BOOLEAN_kw 276
#define DOUBLE_kw 277
#define ANY_kw 278
#define IFCONFIG_kw 279
#define SELECT_kw 280
#define SET_kw 281
#define CONSOLIDATE_kw 282
#define OUTPUT_kw 283
#define STATISTICS_kw 284
#define CLEAR_kw 285
#define TIMER_kw 286
#define HELP_kw 287

/* Value type.  */



int yyparse (void);

#endif /* !YY_YY_SFF_PARSER_HH_INCLUDED  */

/* Copy the second part of user declarations.  */

#line 555 "sff_parser.cc" /* yacc.c:358  */

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif

#ifndef YY_ATTRIBUTE
# if (defined __GNUC__                                               \
      && (2 < __GNUC__ || (__GNUC__ == 2 && 96 <= __GNUC_MINOR__)))  \
     || defined __SUNPRO_C && 0x5110 <= __SUNPRO_C
#  define YY_ATTRIBUTE(Spec) __attribute__(Spec)
# else
#  define YY_ATTRIBUTE(Spec) /* empty */
# endif
#endif

#ifndef YY_ATTRIBUTE_PURE
# define YY_ATTRIBUTE_PURE   YY_ATTRIBUTE ((__pure__))
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# define YY_ATTRIBUTE_UNUSED YY_ATTRIBUTE ((__unused__))
#endif

#if !defined _Noreturn \
     && (!defined __STDC_VERSION__ || __STDC_VERSION__ < 201112)
# if defined _MSC_VER && 1200 <= _MSC_VER
#  define _Noreturn __declspec (noreturn)
# else
#  define _Noreturn YY_ATTRIBUTE ((__noreturn__))
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif

#if defined __GNUC__ && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN \
    _Pragma ("GCC diagnostic push") \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")\
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define YY_IGNORE_MAYBE_UNINITIALIZED_END \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif


#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYSIZE_T yynewbytes;                                            \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / sizeof (*yyptr);                          \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, (Count) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYSIZE_T yyi;                         \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  50
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   178

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  37
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  23
/* YYNRULES -- Number of rules.  */
#define YYNRULES  83
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  116

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   287

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,    34,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,    33,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    35,     2,    36,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   434,   434,   435,   436,   439,   440,   441,   442,   443,
     444,   445,   446,   447,   450,   454,   461,   471,   486,   538,
     551,   557,   566,   569,   570,   582,   596,   601,   607,   640,
     674,   675,   676,   677,   678,   679,   680,   681,   682,   683,
     684,   685,   686,   687,   688,   689,   692,   703,   712,   725,
     726,   730,   734,   738,   742,   747,   754,   760,   764,   770,
     776,   784,   785,   786,   787,   788,   789,   790,   793,   794,
     795,   796,   797,   801,   808,   817,   827,   837,   837,   840,
     843,   884,   927,   936
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || 0
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "ID_v", "STR_v", "REGEX_V", "INT_v",
  "BOOL_v", "DOUBLE_v", "AND_op", "OR_op", "LT_op", "GT_op", "EQ_op",
  "NE_op", "PF_op", "SF_op", "SS_op", "RE_op", "INTEGER_kw", "STRING_kw",
  "BOOLEAN_kw", "DOUBLE_kw", "ANY_kw", "IFCONFIG_kw", "SELECT_kw",
  "SET_kw", "CONSOLIDATE_kw", "OUTPUT_kw", "STATISTICS_kw", "CLEAR_kw",
  "TIMER_kw", "HELP_kw", "';'", "'-'", "'{'", "'}'", "$accept", "StmtList",
  "Stmt", "Help", "SetParameter", "Timer", "Statistics", "Consolidate",
  "Clear", "Output", "IfConfig", "Id", "Predicate", "Constraint",
  "AnyConstraint", "Op", "Value", "TagSetList", "TagSet", "opt_Comma",
  "Tag", "Select", "Message", YY_NULLPTR
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,    59,    45,   123,   125
};
# endif

#define YYPACT_NINF -92

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-92)))

#define YYTABLE_NINF -79

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
      72,   -29,    15,     5,   132,   -92,   102,    19,   132,   132,
     132,    54,   -23,   -92,   -92,   -92,   -92,   -92,   -92,   -92,
     -92,   -92,   -92,    40,   -92,   -92,   -92,   -92,   -92,   -92,
     -92,   -92,   -92,   -92,   -92,   -92,   -92,   -92,   -92,   -92,
     132,    25,   132,    42,     7,   -92,   -92,   -92,   -92,   -92,
     -92,    -5,   -92,   132,   155,     4,    13,   -92,     6,   -92,
      84,    43,   105,   -92,   -92,   -92,     9,   -92,   -92,   -92,
     -92,   -92,   -92,   -92,    52,    30,   -92,   -92,   142,   132,
     132,   -92,    28,   -92,   132,   -92,   -92,   -92,   -92,   -92,
     -92,    84,   -92,   -92,   -92,   -92,   -92,   -92,   -92,   -92,
     -92,   -92,   -92,   -92,   -92,   -92,   -92,   155,   155,   132,
     -92,   -92,   -92,   -92,    10,   -92
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       0,     0,     0,     0,     0,    22,     0,    20,    23,     0,
      14,     0,     0,    13,    12,    11,    10,     9,     8,     7,
       5,     6,     4,     0,    30,    45,    44,    31,    32,    33,
      34,    35,    36,    37,    38,    39,    42,    43,    40,    41,
       0,     0,    80,     0,     0,    25,    21,    24,    19,    15,
       1,     0,     2,     0,     0,    28,    29,    79,    78,    75,
       0,     0,     0,    26,    27,     3,    78,    61,    62,    63,
      64,    65,    66,    67,     0,     0,    46,    49,     0,     0,
       0,    77,     0,    81,     0,    72,    68,    70,    69,    71,
      82,     0,    18,    17,    16,    73,    55,    56,    60,    59,
      58,    57,    54,    50,    53,    52,    51,     0,     0,     0,
      76,    83,    47,    48,    78,    74
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int8 yypgoto[] =
{
     -92,   -92,    63,   -92,   -92,   -92,   -92,   -92,   -92,   -92,
     -92,    -3,   -92,   -91,   -92,   -92,    -4,   -92,   -51,    33,
      11,   -92,   -92
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int8 yydefgoto[] =
{
      -1,    11,    12,    13,    14,    15,    16,    17,    18,    19,
      20,    57,    55,    76,    77,    78,    90,    56,    58,    84,
      59,    21,    42
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int8 yytable[] =
{
      41,    43,    66,    45,    22,    47,    48,    49,    24,    25,
      52,    63,    26,    79,    80,    81,   112,   113,    81,    81,
      54,    23,    81,    46,    27,    28,    29,    30,    65,    31,
      32,    33,    34,    35,    36,    37,    38,    39,    60,    61,
      40,    64,    83,    24,    25,    95,   115,    26,   -78,    97,
      98,    99,   100,   101,    50,    62,    91,    96,   114,    27,
      28,    29,    30,   109,    31,    32,    33,    34,    35,    36,
      37,    38,    39,     1,    51,    53,   107,   108,     2,     3,
       4,     5,     6,     7,     8,     9,    10,   111,    85,    82,
      86,    87,    88,     0,     0,   110,     2,     3,     4,     5,
       6,     7,     8,     9,    10,    24,    25,    89,    92,    26,
       0,    93,    94,     0,    44,     0,     0,     0,     0,     0,
       0,    27,    28,    29,    30,     0,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    24,    25,     0,     0,    26,
       0,     0,     0,     0,     0,     0,   102,     0,   103,   104,
     105,    27,    28,    29,    30,     0,    31,    32,    33,    34,
      35,    36,    37,    38,    39,   106,    67,    68,    69,    70,
      71,    72,    73,    74,     0,     0,     0,     0,    75
};

static const yytype_int8 yycheck[] =
{
       3,     4,    53,     6,    33,     8,     9,    10,     3,     4,
      33,     4,     7,     9,    10,     9,   107,   108,     9,     9,
      23,     6,     9,     4,    19,    20,    21,    22,    33,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    13,    42,
      35,    34,    36,     3,     4,    36,    36,     7,    35,    19,
      20,    21,    22,    23,     0,    13,    13,     5,   109,    19,
      20,    21,    22,    35,    24,    25,    26,    27,    28,    29,
      30,    31,    32,     1,    11,    35,    79,    80,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    91,     4,    56,
       6,     7,     8,    -1,    -1,    84,    24,    25,    26,    27,
      28,    29,    30,    31,    32,     3,     4,    23,     3,     7,
      -1,     6,     7,    -1,    12,    -1,    -1,    -1,    -1,    -1,
      -1,    19,    20,    21,    22,    -1,    24,    25,    26,    27,
      28,    29,    30,    31,    32,     3,     4,    -1,    -1,     7,
      -1,    -1,    -1,    -1,    -1,    -1,     4,    -1,     6,     7,
       8,    19,    20,    21,    22,    -1,    24,    25,    26,    27,
      28,    29,    30,    31,    32,    23,    11,    12,    13,    14,
      15,    16,    17,    18,    -1,    -1,    -1,    -1,    23
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     1,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    38,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    58,    33,     6,     3,     4,     7,    19,    20,    21,
      22,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      35,    48,    59,    48,    12,    48,     4,    48,    48,    48,
       0,    39,    33,    35,    48,    49,    54,    48,    55,    57,
      13,    48,    13,     4,    34,    33,    55,    11,    12,    13,
      14,    15,    16,    17,    18,    23,    50,    51,    52,     9,
      10,     9,    56,    36,    56,     4,     6,     7,     8,    23,
      53,    13,     3,     6,     7,    36,     5,    19,    20,    21,
      22,    23,     4,     6,     7,     8,    23,    48,    48,    35,
      57,    53,    50,    50,    55,    36
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    37,    38,    38,    38,    39,    39,    39,    39,    39,
      39,    39,    39,    39,    40,    40,    41,    41,    41,    42,
      43,    43,    44,    45,    45,    46,    46,    46,    47,    47,
      48,    48,    48,    48,    48,    48,    48,    48,    48,    48,
      48,    48,    48,    48,    48,    48,    49,    49,    49,    50,
      50,    50,    50,    50,    50,    50,    51,    51,    51,    51,
      51,    52,    52,    52,    52,    52,    52,    52,    53,    53,
      53,    53,    53,    54,    54,    55,    55,    56,    56,    57,
      58,    58,    59,    59
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     2,     3,     2,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     2,     4,     4,     4,     2,
       1,     2,     1,     1,     2,     2,     3,     3,     3,     3,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     2,     4,     4,     1,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     3,     5,     1,     3,     1,     0,     1,
       2,     4,     3,     4
};


#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)
#define YYEMPTY         (-2)
#define YYEOF           0

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                  \
do                                                              \
  if (yychar == YYEMPTY)                                        \
    {                                                           \
      yychar = (Token);                                         \
      yylval = (Value);                                         \
      YYPOPSTACK (yylen);                                       \
      yystate = *yyssp;                                         \
      goto yybackup;                                            \
    }                                                           \
  else                                                          \
    {                                                           \
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;                                                  \
    }                                                           \
while (0)

/* Error token number */
#define YYTERROR        1
#define YYERRCODE       256



/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)

/* This macro is provided for backward compatibility. */
#ifndef YY_LOCATION_PRINT
# define YY_LOCATION_PRINT(File, Loc) ((void) 0)
#endif


# define YY_SYMBOL_PRINT(Title, Type, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Type, Value); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*----------------------------------------.
| Print this symbol's value on YYOUTPUT.  |
`----------------------------------------*/

static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
{
  FILE *yyo = yyoutput;
  YYUSE (yyo);
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# endif
  YYUSE (yytype);
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
{
  YYFPRINTF (yyoutput, "%s %s (",
             yytype < YYNTOKENS ? "token" : "nterm", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yytype_int16 *yyssp, YYSTYPE *yyvsp, int yyrule)
{
  unsigned long int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       yystos[yyssp[yyi + 1 - yynrhs]],
                       &(yyvsp[(yyi + 1) - (yynrhs)])
                                              );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif


#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
static YYSIZE_T
yystrlen (const char *yystr)
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
yystpcpy (char *yydest, const char *yysrc)
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
        switch (*++yyp)
          {
          case '\'':
          case ',':
            goto do_not_strip_quotes;

          case '\\':
            if (*++yyp != '\\')
              goto do_not_strip_quotes;
            /* Fall through.  */
          default:
            if (yyres)
              yyres[yyn] = *yyp;
            yyn++;
            break;

          case '"':
            if (yyres)
              yyres[yyn] = '\0';
            return yyn;
          }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return 1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store.  */
static int
yysyntax_error (YYSIZE_T *yymsg_alloc, char **yymsg,
                yytype_int16 *yyssp, int yytoken)
{
  YYSIZE_T yysize0 = yytnamerr (YY_NULLPTR, yytname[yytoken]);
  YYSIZE_T yysize = yysize0;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULLPTR;
  /* Arguments of yyformat. */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Number of reported tokens (one for the "unexpected", one per
     "expected"). */
  int yycount = 0;

  /* There are many possibilities here to consider:
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yytoken != YYEMPTY)
    {
      int yyn = yypact[*yyssp];
      yyarg[yycount++] = yytname[yytoken];
      if (!yypact_value_is_default (yyn))
        {
          /* Start YYX at -YYN if negative to avoid negative indexes in
             YYCHECK.  In other words, skip the first -YYN actions for
             this state because they are default actions.  */
          int yyxbegin = yyn < 0 ? -yyn : 0;
          /* Stay within bounds of both yycheck and yytname.  */
          int yychecklim = YYLAST - yyn + 1;
          int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
          int yyx;

          for (yyx = yyxbegin; yyx < yyxend; ++yyx)
            if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR
                && !yytable_value_is_error (yytable[yyx + yyn]))
              {
                if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
                  {
                    yycount = 1;
                    yysize = yysize0;
                    break;
                  }
                yyarg[yycount++] = yytname[yyx];
                {
                  YYSIZE_T yysize1 = yysize + yytnamerr (YY_NULLPTR, yytname[yyx]);
                  if (! (yysize <= yysize1
                         && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
                    return 2;
                  yysize = yysize1;
                }
              }
        }
    }

  switch (yycount)
    {
# define YYCASE_(N, S)                      \
      case N:                               \
        yyformat = S;                       \
      break
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
# undef YYCASE_
    }

  {
    YYSIZE_T yysize1 = yysize + yystrlen (yyformat);
    if (! (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
      return 2;
    yysize = yysize1;
  }

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return 1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
        {
          yyp += yytnamerr (yyp, yyarg[yyi++]);
          yyformat += 2;
        }
      else
        {
          yyp++;
          yyformat++;
        }
  }
  return 0;
}
#endif /* YYERROR_VERBOSE */

/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
{
  YYUSE (yyvaluep);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE (yytype);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}




/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
/* The lookahead symbol.  */
int yychar;


/* The semantic value of the lookahead symbol.  */
/* Default value used for initialization, for pacifying older GCCs
   or non-GCC compilers.  */
YY_INITIAL_VALUE (static YYSTYPE yyval_default;)
YYSTYPE yylval YY_INITIAL_VALUE (= yyval_default);

    /* Number of syntax errors so far.  */
    int yynerrs;

    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       'yyss': related to states.
       'yyvs': related to semantic values.

       Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken = 0;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yyssp = yyss = yyssa;
  yyvsp = yyvs = yyvsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */
  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        YYSTYPE *yyvs1 = yyvs;
        yytype_int16 *yyss1 = yyss;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * sizeof (*yyssp),
                    &yyvs1, yysize * sizeof (*yyvsp),
                    &yystacksize);

        yyss = yyss1;
        yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yytype_int16 *yyss1 = yyss;
        union yyalloc *yyptr =
          (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
        if (! yyptr)
          goto yyexhaustedlab;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
                  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = yylex (&yylval);
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
  yychar = YYEMPTY;

  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 2:
#line 434 "sff_parser.yy" /* yacc.c:1646  */
    { sff_parser_complete_command(); }
#line 1733 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 3:
#line 435 "sff_parser.yy" /* yacc.c:1646  */
    { sff_parser_complete_command(); }
#line 1739 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 4:
#line 436 "sff_parser.yy" /* yacc.c:1646  */
    { sff_parser_complete_command(); }
#line 1745 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 14:
#line 451 "sff_parser.yy" /* yacc.c:1646  */
    {
       sff_parser_help(0);
   }
#line 1753 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 15:
#line 455 "sff_parser.yy" /* yacc.c:1646  */
    {
       sff_parser_help((yyvsp[0].str_v)->c_str());
       delete((yyvsp[0].str_v));
   }
#line 1762 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 16:
#line 462 "sff_parser.yy" /* yacc.c:1646  */
    {
       if (*(yyvsp[-2].str_v) == "statistics_only" || *(yyvsp[-2].str_v) == "stats_only") {
	   stats_only = (yyvsp[0].bool_v);
       } else {
	   std::cerr << yysfffname << ':' << (yyvsp[-3].nlin) // implicitly nlin
		     << ": unknown Boolean parameter " << *(yyvsp[-2].str_v) << std::endl;
       }
       delete((yyvsp[-2].str_v));
   }
#line 1776 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 17:
#line 472 "sff_parser.yy" /* yacc.c:1646  */
    {
       if (*(yyvsp[-2].str_v) == "preprocessing_limit") {
	   siena::FwdTable * FwdT = dynamic_cast<siena::FwdTable *>(FT);
	   if (FwdT) {
	       FwdT->set_preprocess_rounds((yyvsp[0].int_v)); 
	   } else {
	       std::cerr << "Not using the FwdTable algorithm." << std::endl;
	   }
       } else {
	   std::cerr << yysfffname << ':' << (yyvsp[-3].nlin) // implicitly nlin
		     << ": unknown integer parameter " << *(yyvsp[-2].str_v) << std::endl;
       }
       delete((yyvsp[-2].str_v));
   }
#line 1795 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 18:
#line 487 "sff_parser.yy" /* yacc.c:1646  */
    {
       if (*(yyvsp[-2].str_v) == "algorithm") {
	   if (*(yyvsp[0].str_v) == "fwdtable" || *(yyvsp[0].str_v) == "FwdTable") {
	       sff_parser_use_fwdtable();
	   }
#ifdef HAVE_CUDD
	   else if (*(yyvsp[0].str_v) == "bddbtable" || *(yyvsp[0].str_v) == "BDDBTable") {
	       sff_parser_use_bddbtable();
	   }
	   else if (*(yyvsp[0].str_v) == "zddbtable" || *(yyvsp[0].str_v) == "ZDDBTable") {
	       sff_parser_use_zddbtable();
	   }
#endif
	   else if (*(yyvsp[0].str_v) == "btable" || *(yyvsp[0].str_v) == "BTable") {
	       sff_parser_use_btable();
	   }
	   else if (*(yyvsp[0].str_v) == "btrietable" || *(yyvsp[0].str_v) == "BTrieTable") {
	       sff_parser_use_btrietable();
	   }
	   else if (*(yyvsp[0].str_v) == "sorted_btable" || *(yyvsp[0].str_v) == "SortedBTable") {
	       sff_parser_use_sorted_btable();
	   }
	   else if (*(yyvsp[0].str_v) == "bxtable" || *(yyvsp[0].str_v) == "BXTable") {
	       sff_parser_use_bxtable();
	   }
	   else if (*(yyvsp[0].str_v) == "bctable" || *(yyvsp[0].str_v) == "BCTable") {
	       sff_parser_use_bctable();
	   }
	   else if (*(yyvsp[0].str_v) == "bvtable" || *(yyvsp[0].str_v) == "BVTable") {
	       sff_parser_use_bvtable();
	   }
	   else if (*(yyvsp[0].str_v) == "ttable" || *(yyvsp[0].str_v) == "TTable") {
	       sff_parser_use_ttable();
	   }
	   else if (*(yyvsp[0].str_v) == "tagstable" || *(yyvsp[0].str_v) == "TagsTable") {
	       sff_parser_use_tagstable();
	   }
	   else {
	   std::cerr << yysfffname << ':' << (yyvsp[-3].nlin) // implicitly nlin
		     << ": unknown algorithm " << *(yyvsp[0].str_v) << std::endl;
	   }
       } else {
	   std::cerr << yysfffname << ':' << (yyvsp[-3].nlin) // implicitly nlin
		     << ": unknown string parameter " << *(yyvsp[-2].str_v) << std::endl;
       }
       delete((yyvsp[-2].str_v));
       delete((yyvsp[0].str_v));
   }
#line 1848 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 19:
#line 539 "sff_parser.yy" /* yacc.c:1646  */
    {
       if (*(yyvsp[0].str_v) == "start") {
	   sff_parser_timer_start();
       } else if (*(yyvsp[0].str_v) == "stop") {
	   sff_parser_timer_stop();
       } else {
	   std::cerr << yysfffname << ':' << (yyvsp[-1].nlin) // implicitly nlin
		     << ": unknown parameter " << *(yyvsp[0].str_v) << std::endl;
       }
       delete((yyvsp[0].str_v));
   }
#line 1864 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 20:
#line 552 "sff_parser.yy" /* yacc.c:1646  */
    { 
       if (errors) 
	   std::cerr << "Warning: there were errors in the input." << std::endl;
       sff_parser_print_statistics(StatsFormat);
   }
#line 1874 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 21:
#line 558 "sff_parser.yy" /* yacc.c:1646  */
    { 
       if (errors) 
	   std::cerr << "Warning: there were errors in the input." << std::endl;
       sff_parser_print_statistics(((yyvsp[0].str_v)->length() > 0) ? ((yyvsp[0].str_v)->c_str()) : StatsFormat); 
       delete((yyvsp[0].str_v));
   }
#line 1885 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 22:
#line 566 "sff_parser.yy" /* yacc.c:1646  */
    { sff_consolidate(); }
#line 1891 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 23:
#line 569 "sff_parser.yy" /* yacc.c:1646  */
    { sff_clear(); }
#line 1897 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 24:
#line 571 "sff_parser.yy" /* yacc.c:1646  */
    {
       if (*(yyvsp[0].str_v) == "recycle" || *(yyvsp[0].str_v) == "-r" || *(yyvsp[0].str_v) == "--recycle") {
	   sff_clear_recycle();
       } else {
	   std::cerr << yysfffname << ':' << (yyvsp[-1].nlin) // implicitly nlin
		     << ": unknown parameter " << *(yyvsp[0].str_v) << std::endl;
       }
       delete((yyvsp[0].str_v));
    }
#line 1911 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 25:
#line 583 "sff_parser.yy" /* yacc.c:1646  */
    {
       if (*(yyvsp[0].str_v) == "on") {
	   output_level = SFF_VERBOSE;
       } else if (*(yyvsp[0].str_v) == "off") {
	   output_level = SFF_SILENT; 
       } else if (*(yyvsp[0].str_v) == "count") {
	   output_level = SFF_MATCH_COUNT; 
       } else {
	   std::cerr << yysfffname << ':' << (yyvsp[-1].nlin) // implicitly nlin
		     << ": unknown parameter " << *(yyvsp[0].str_v) << std::endl;
       }
       delete((yyvsp[0].str_v));
   }
#line 1929 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 26:
#line 597 "sff_parser.yy" /* yacc.c:1646  */
    {
       sff_parser_open_output((yyvsp[0].str_v)->c_str()); 
       delete((yyvsp[0].str_v));
   }
#line 1938 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 27:
#line 602 "sff_parser.yy" /* yacc.c:1646  */
    { 
       sff_parser_open_output(0); 
   }
#line 1946 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 28:
#line 608 "sff_parser.yy" /* yacc.c:1646  */
    {
       if (! stats_only) {
	   if (ifconfig_ranges && !(*ifconfig_ranges)[(yyvsp[-1].int_v)])
	       goto cleanup_ifat;
	    if (!FT) {
		if (TT) {
		    std::cerr << yysfffname << ':' << (yyvsp[-2].nlin) // implicitly nlin
			      << ": tags table already active" << std::endl;
		    goto cleanup_ifat;
		} else {
		    // default FT:
		    sff_parser_use_fwdtable();
		}
	    } 
	    if (consolidate_guard) {
		std::cerr << yysfffname << ':' << (yyvsp[-2].nlin) // implicitly nlin
			  << ": forwarding table already consolidated" << std::endl;
	    } else {
		try {
		    FT->ifconfig((yyvsp[-1].int_v), *(yyvsp[0].predicate));
		} catch (siena::BadConstraint & ex) {
		    std::cerr << yysfffname << ':' << (yyvsp[-2].nlin) // implicitly nlin
			      << ": bad constraint in predicate: " 
			      << ex.what() << std::endl;
		}
	    }
	cleanup_ifat:
	    Mem.recycle();
	    delete((yyvsp[0].predicate));
	}
	++i_counter;
    }
#line 1983 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 29:
#line 641 "sff_parser.yy" /* yacc.c:1646  */
    {
	if (! stats_only) {
	    if (ifconfig_ranges && !(*ifconfig_ranges)[(yyvsp[-1].int_v)])
		goto cleanup_iftt;
	    if (!TT) {
		if (FT) {
		    std::cerr << yysfffname << ':' << (yyvsp[-2].nlin) // implicitly nlin
			      << ": forwarding table already active" << std::endl;
		    goto cleanup_iftt;
		} else {
		    // default TT:
		    sff_parser_use_tagstable();
		}
	    } 
	    if (consolidate_guard) {
		std::cerr << yysfffname << ':' << (yyvsp[-2].nlin) // implicitly nlin
			  << ": tags table already consolidated" << std::endl;
	    } 
	    try {
		TT->ifconfig((yyvsp[-1].int_v), *(yyvsp[0].tagset_list));
	    } catch (siena::BadConstraint & ex) {
		std::cerr << yysfffname << ':' << (yyvsp[-2].nlin) // implicitly nlin
			  << ": bad constraint in predicate: " 
			  << ex.what() << std::endl;
	    }
	cleanup_iftt:
	    Mem.recycle();
	    delete((yyvsp[0].tagset_list));
	}
	++i_counter;
    }
#line 2019 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 30:
#line 674 "sff_parser.yy" /* yacc.c:1646  */
    { (yyval.str_v) = (yyvsp[0].str_v); }
#line 2025 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 31:
#line 675 "sff_parser.yy" /* yacc.c:1646  */
    { (yyval.str_v) = new std::string("integer"); }
#line 2031 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 32:
#line 676 "sff_parser.yy" /* yacc.c:1646  */
    { (yyval.str_v) = new std::string("string"); }
#line 2037 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 33:
#line 677 "sff_parser.yy" /* yacc.c:1646  */
    { (yyval.str_v) = new std::string("boolean"); }
#line 2043 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 34:
#line 678 "sff_parser.yy" /* yacc.c:1646  */
    { (yyval.str_v) = new std::string("double"); }
#line 2049 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 35:
#line 679 "sff_parser.yy" /* yacc.c:1646  */
    { (yyval.str_v) = new std::string("ifconfig"); }
#line 2055 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 36:
#line 680 "sff_parser.yy" /* yacc.c:1646  */
    { (yyval.str_v) = new std::string("select"); }
#line 2061 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 37:
#line 681 "sff_parser.yy" /* yacc.c:1646  */
    { (yyval.str_v) = new std::string("set"); }
#line 2067 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 38:
#line 682 "sff_parser.yy" /* yacc.c:1646  */
    { (yyval.str_v) = new std::string("consolidate"); }
#line 2073 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 39:
#line 683 "sff_parser.yy" /* yacc.c:1646  */
    { (yyval.str_v) = new std::string("output"); }
#line 2079 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 40:
#line 684 "sff_parser.yy" /* yacc.c:1646  */
    { (yyval.str_v) = new std::string("timer"); }
#line 2085 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 41:
#line 685 "sff_parser.yy" /* yacc.c:1646  */
    { (yyval.str_v) = new std::string("help"); }
#line 2091 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 42:
#line 686 "sff_parser.yy" /* yacc.c:1646  */
    { (yyval.str_v) = new std::string("statistics"); }
#line 2097 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 43:
#line 687 "sff_parser.yy" /* yacc.c:1646  */
    { (yyval.str_v) = new std::string("clear"); }
#line 2103 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 44:
#line 688 "sff_parser.yy" /* yacc.c:1646  */
    { (yyval.str_v) = ((yyvsp[0].bool_v)) ? (new std::string("true")) : (new std::string("false")); }
#line 2109 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 45:
#line 689 "sff_parser.yy" /* yacc.c:1646  */
    { (yyval.str_v) = (yyvsp[0].str_v); }
#line 2115 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 46:
#line 693 "sff_parser.yy" /* yacc.c:1646  */
    {
	if (! stats_only) {
	    (yyval.predicate) = new simple_predicate();
	    (yyval.predicate)->add(new simple_filter());
	    (yyval.predicate)->last_filter()->add(new_sx_string((yyvsp[-1].str_v)), (yyvsp[0].constraint));
	}
	++f_counter;
	++c_counter;
	delete((yyvsp[-1].str_v));
    }
#line 2130 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 47:
#line 704 "sff_parser.yy" /* yacc.c:1646  */
    {
	if (! stats_only) {
	    (yyval.predicate) = (yyvsp[-3].predicate);
	    (yyval.predicate)->last_filter()->add(new_sx_string((yyvsp[-1].str_v)), (yyvsp[0].constraint));
	}
	++c_counter;
	delete((yyvsp[-1].str_v));
    }
#line 2143 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 48:
#line 713 "sff_parser.yy" /* yacc.c:1646  */
    {
	if (! stats_only) {
	    (yyval.predicate) = (yyvsp[-3].predicate);
	    (yyval.predicate)->add(new simple_filter());
	    (yyval.predicate)->last_filter()->add(new_sx_string((yyvsp[-1].str_v)), (yyvsp[0].constraint));
	}
	++f_counter;
	++c_counter;
	delete((yyvsp[-1].str_v));
    }
#line 2158 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 49:
#line 725 "sff_parser.yy" /* yacc.c:1646  */
    { if (!stats_only) { (yyval.constraint) = (yyvsp[0].constraint); } }
#line 2164 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 50:
#line 727 "sff_parser.yy" /* yacc.c:1646  */
    { 
	if (!stats_only) { (yyval.constraint) = new simple_op_value((yyvsp[-1].op), (yyvsp[0].int_v)); } 
    }
#line 2172 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 51:
#line 731 "sff_parser.yy" /* yacc.c:1646  */
    { 
	if (!stats_only) { (yyval.constraint) = new simple_op_value((yyvsp[-1].op)); } 
    }
#line 2180 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 52:
#line 735 "sff_parser.yy" /* yacc.c:1646  */
    { 
	if (!stats_only) { (yyval.constraint) = new simple_op_value((yyvsp[-1].op), (yyvsp[0].double_v)); } 
    }
#line 2188 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 53:
#line 739 "sff_parser.yy" /* yacc.c:1646  */
    { 
	if (!stats_only) { (yyval.constraint) = new simple_op_value((yyvsp[-1].op), (yyvsp[0].bool_v)); } 
    }
#line 2196 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 54:
#line 743 "sff_parser.yy" /* yacc.c:1646  */
    { 
	if (!stats_only) { (yyval.constraint) = new simple_op_value((yyvsp[-1].op), new_sx_string((yyvsp[0].str_v))); }
	delete((yyvsp[0].str_v)); 
    }
#line 2205 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 55:
#line 748 "sff_parser.yy" /* yacc.c:1646  */
    {
	if (!stats_only) { (yyval.constraint) = new simple_op_value(siena::RE, new_sx_string((yyvsp[0].str_v))); }
	delete((yyvsp[0].str_v)); 
    }
#line 2214 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 56:
#line 755 "sff_parser.yy" /* yacc.c:1646  */
    { 
	if (!stats_only) { 
	    (yyval.constraint) = new simple_op_value(siena::ANY, (siena::Int)0); 
	} 
    }
#line 2224 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 57:
#line 761 "sff_parser.yy" /* yacc.c:1646  */
    { 
	if (!stats_only) { (yyval.constraint) = new simple_op_value(siena::ANY); } 
    }
#line 2232 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 58:
#line 765 "sff_parser.yy" /* yacc.c:1646  */
    { 
	if (!stats_only) { 
	    (yyval.constraint) = new simple_op_value(siena::ANY, (siena::Double).0);
	} 
    }
#line 2242 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 59:
#line 771 "sff_parser.yy" /* yacc.c:1646  */
    { 
	if (!stats_only) { 
	    (yyval.constraint) = new simple_op_value(siena::ANY, false);
	} 
    }
#line 2252 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 60:
#line 777 "sff_parser.yy" /* yacc.c:1646  */
    { 
	if (!stats_only) { 
	    (yyval.constraint) = new simple_op_value(siena::ANY, new_sx_string()); 
	} 
    }
#line 2262 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 61:
#line 784 "sff_parser.yy" /* yacc.c:1646  */
    { (yyval.op) = siena::LT; }
#line 2268 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 62:
#line 785 "sff_parser.yy" /* yacc.c:1646  */
    { (yyval.op) = siena::GT; }
#line 2274 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 63:
#line 786 "sff_parser.yy" /* yacc.c:1646  */
    { (yyval.op) = siena::EQ; }
#line 2280 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 64:
#line 787 "sff_parser.yy" /* yacc.c:1646  */
    { (yyval.op) = siena::NE; }
#line 2286 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 65:
#line 788 "sff_parser.yy" /* yacc.c:1646  */
    { (yyval.op) = siena::PF; }
#line 2292 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 66:
#line 789 "sff_parser.yy" /* yacc.c:1646  */
    { (yyval.op) = siena::SF; }
#line 2298 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 67:
#line 790 "sff_parser.yy" /* yacc.c:1646  */
    { (yyval.op) = siena::SS; }
#line 2304 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 68:
#line 793 "sff_parser.yy" /* yacc.c:1646  */
    { if (!stats_only) { (yyval.value) = new simple_value((yyvsp[0].int_v)); } }
#line 2310 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 69:
#line 794 "sff_parser.yy" /* yacc.c:1646  */
    { if (!stats_only) { (yyval.value) = new simple_value((yyvsp[0].double_v)); } }
#line 2316 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 70:
#line 795 "sff_parser.yy" /* yacc.c:1646  */
    { if (!stats_only) { (yyval.value) = new simple_value((yyvsp[0].bool_v)); } }
#line 2322 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 71:
#line 796 "sff_parser.yy" /* yacc.c:1646  */
    { if (!stats_only) { (yyval.value) = new simple_value(); } }
#line 2328 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 72:
#line 797 "sff_parser.yy" /* yacc.c:1646  */
    { if (!stats_only) { (yyval.value) = new simple_value(new_sx_string((yyvsp[0].str_v))); }
		delete((yyvsp[0].str_v)); }
#line 2335 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 73:
#line 802 "sff_parser.yy" /* yacc.c:1646  */
    {
	if (!stats_only) { 
	    (yyval.tagset_list) = new simple_tagset_list();
	    (yyval.tagset_list)->add_tagset((yyvsp[-1].tagset));
	}
    }
#line 2346 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 74:
#line 809 "sff_parser.yy" /* yacc.c:1646  */
    {
	if (!stats_only) { 
	    (yyval.tagset_list) = (yyvsp[-4].tagset_list);
	    (yyval.tagset_list)->add_tagset((yyvsp[-1].tagset));
	}
    }
#line 2357 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 75:
#line 818 "sff_parser.yy" /* yacc.c:1646  */
    {
	if (!stats_only) { 
	    (yyval.tagset) = new simple_tagset();
	    (yyval.tagset)->add_tag(*((yyvsp[0].str_v)));
	    delete((yyvsp[0].str_v));
	}
	++ts_counter;
	++t_counter;
    }
#line 2371 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 76:
#line 828 "sff_parser.yy" /* yacc.c:1646  */
    {
	if (!stats_only) { 
	    (yyval.tagset) = (yyvsp[-2].tagset);
	    (yyval.tagset)->add_tag(*((yyvsp[0].str_v)));
	    delete((yyvsp[0].str_v));
	}
	++t_counter;
    }
#line 2384 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 79:
#line 840 "sff_parser.yy" /* yacc.c:1646  */
    { (yyval.str_v) = (yyvsp[0].str_v); }
#line 2390 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 80:
#line 844 "sff_parser.yy" /* yacc.c:1646  */
    {
	if (! stats_only) {
	    if (!FT) {
		std::cerr << yysfffname << ':' << (yyvsp[-1].nlin) // implicitly nlin
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

		    FT->match(*(yyvsp[0].message), cc);

		    if (cc.get_count() > 0) ++x_counter;
		    xx_counter += cc.get_count();
		    if (output_level == SFF_MATCH_COUNT) {
			(*sff_output) << cc.get_count() << std::endl;
		    }
		    break;
		}
		case SFF_VERBOSE: {    
		    IPrinter ccc;

		    FT->match(*(yyvsp[0].message), ccc);

		    if (ccc.get_count() > 0) ++x_counter;
		    xx_counter += ccc.get_count();
		    ccc.flush(*sff_output);
		}
		}
	    }
	    Mem.recycle();
	    delete((yyvsp[0].message));
	}
	++m_counter;
    }
#line 2435 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 81:
#line 885 "sff_parser.yy" /* yacc.c:1646  */
    {
	if (! stats_only) {
	    if (!TT) {
		std::cerr << yysfffname << ':' << (yyvsp[-3].nlin) // implicitly nlin
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

		    TT->match(*(yyvsp[-1].tagset), cc);

		    if (cc.get_count() > 0) ++x_counter;
		    xx_counter += cc.get_count();
		    if (output_level == SFF_MATCH_COUNT) {
			(*sff_output) << cc.get_count() << std::endl;
		    }
		}
		    break;
		case SFF_VERBOSE: {    
		    IPrinter ccc;

		    TT->match(*(yyvsp[-1].tagset), ccc);

		    if (ccc.get_count() > 0) ++x_counter;
		    xx_counter += ccc.get_count();
		    ccc.flush(*sff_output);
		}
		}
	    }
	    Mem.recycle();
	    delete((yyvsp[-1].tagset));
	}
	++m_counter;
    }
#line 2480 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 82:
#line 928 "sff_parser.yy" /* yacc.c:1646  */
    { 
	if (! stats_only) {
	    (yyval.message) = new simple_message();
	    (yyval.message)->add(new_sx_string((yyvsp[-2].str_v)), (yyvsp[0].value));
	}
	++a_counter;
       delete((yyvsp[-2].str_v));        
   }
#line 2493 "sff_parser.cc" /* yacc.c:1646  */
    break;

  case 83:
#line 937 "sff_parser.yy" /* yacc.c:1646  */
    {
       if (! stats_only) {
	   (yyval.message) = (yyvsp[-3].message);
	   (yyval.message)->add(new_sx_string((yyvsp[-2].str_v)), (yyvsp[0].value));
       }
       ++a_counter;
       delete((yyvsp[-2].str_v)); 
   }
#line 2506 "sff_parser.cc" /* yacc.c:1646  */
    break;


#line 2510 "sff_parser.cc" /* yacc.c:1646  */
      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYEMPTY : YYTRANSLATE (yychar);

  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
# define YYSYNTAX_ERROR yysyntax_error (&yymsg_alloc, &yymsg, \
                                        yyssp, yytoken)
      {
        char const *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        yysyntax_error_status = YYSYNTAX_ERROR;
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == 1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = (char *) YYSTACK_ALLOC (yymsg_alloc);
            if (!yymsg)
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = 2;
              }
            else
              {
                yysyntax_error_status = YYSYNTAX_ERROR;
                yymsgp = yymsg;
              }
          }
        yyerror (yymsgp);
        if (yysyntax_error_status == 2)
          goto yyexhaustedlab;
      }
# undef YYSYNTAX_ERROR
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYTERROR;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;


      yydestruct ("Error: popping",
                  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#if !defined yyoverflow || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  return yyresult;
}
#line 947 "sff_parser.yy" /* yacc.c:1906  */


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
