/* config.h.  Generated from config.h.in by configure.  */
/* config.h.in.  Generated from configure.ac by autoheader.  */

/* number of hash functions used in the Bloom filters used in the BTable,
   BCTable, and BXTable algorithms. */
#define CONFIG_BLOOM_FILTER_K 10

/* size of the Bloom filters (bits) used in the BTable, BCTable, and BXTable
   algorithms. */
#define CONFIG_BLOOM_FILTER_SIZE 192

/* Enables debugging output */
/* #undef DEBUG_OUTPUT */

/* Monotonic timer. */
/* #undef GETTIME_CLOCK_ID */

/* define if the Boost library is available */
/* #undef HAVE_BOOST */

/* define if the Boost::Unit_Test_Framework library is available */
/* #undef HAVE_BOOST_UNIT_TEST_FRAMEWORK */

/* uses the compiler's intrinsic __builtin_popcountl. */
#define HAVE_BUILTIN_POPCOUNT 1

/* includes BDD options for Bloom filter matching implementation */
/* #undef HAVE_CUDD */

/* define if the compiler supports basic C++11 syntax */
/* #undef HAVE_CXX11 */

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* Intel rdtsc instruction */
/* #undef HAVE_RDTSC */

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Define to the sub-directory in which libtool stores uninstalled libraries.
   */
#define LT_OBJDIR ".libs/"

/* Disables assertions and other debugging code */
#define NDEBUG 1

/* Name of package */
#define PACKAGE "sff"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "Antonio Carzaniga (firstname.lastname@usi.ch)"

/* Define to the full name of this package. */
#define PACKAGE_NAME "sff"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "sff 2.0.1"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "sff"

/* Define to the home page for this package. */
#define PACKAGE_URL ""

/* Define to the version of this package. */
#define PACKAGE_VERSION "2.0.1"

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Define to 1 if you can safely include both <sys/time.h> and <time.h>. */
#define TIME_WITH_SYS_TIME 1

/* Enables the use of C++ standard bitset */
/* #undef USE_STD_BITSET */

/* Version number of package */
#define VERSION "2.0.1"

/* uses a TST to implement the attributes index. */
/* #undef WITH_A_INDEX_USING_TST */

/* Using C++-11 std::chrono feature. */
/* #undef WITH_CXX_CHRONO_TIMERS */

/* Using gprof, so do not mess with SIGPROF */
/* #undef WITH_PROFILING */

/* implementing timers with RDTSC */
/* #undef WITH_RDTSC_TIMERS */

/* uses static, faster but non-reentrant counting algorithm. */
/* #undef WITH_STATIC_COUNTERS */

/* libsff maintains per-module performance timers */
/* #undef WITH_TIMERS */

/* Define to 1 if `lex' declares `yytext' as a `char *' by default, not a
   `char[]'. */
/* #undef YYTEXT_POINTER */
