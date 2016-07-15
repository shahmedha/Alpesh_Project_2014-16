#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef WITH_TIMERS

#include <sys/time.h>
#include <sys/resource.h>

#include <climits>

#include "timing.h"
#endif

#define BOOST_TEST_MODULE timers
#define BOOST_TEST_DYN_LINK 1

#include <boost/test/unit_test.hpp>

#ifdef WITH_TIMERS

using namespace std;

//
// here are a couple of mutually recursive, apparently different but
// in fact identical Fibonacci implementations.  These two functions
// are intended to consistently waste some cycles.  In other words, we
// need to defeat the smart compiler optimizations.  Not sure this is
// the best way to do this, but it seems to work.
// 
static unsigned long F2(unsigned long n);

static unsigned long F(unsigned long n) {
    if (n < 2)
	return n;
    else
	return F(n-1) + F2(n-2);
}

static unsigned long F2(unsigned long n) {
    if (n < 2)
	return n;
    else
	return F2(n-1) + F(n-2);
}

static unsigned long read_current_utime () {
    struct rusage ru;
    getrusage (RUSAGE_SELF, &ru);
    return (ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) * 1000000UL 
	+ ru.ru_utime.tv_usec + ru.ru_stime.tv_usec;
}

static unsigned long percent_error(unsigned long t_r, unsigned long t_t) {
    // t_r is assumed to be the TRUE answer, and therefore the percent
    // error is computed as e = |t_r - t_t|/t_r and is reported as an
    // integer in %, that is, multiplied by 100
    if (t_r == 0)
	return (t_t == 0) ? 0 : ULONG_MAX;
    unsigned long e = (t_r > t_t) ? (t_r - t_t) : (t_t - t_r);
    e *= 100;
    e /= t_r;
    return e;
}

static const char * warning_msg = 
    "This test might fail due to a particularly bad scheduling\n"
    "of the test program.  You might want to re-run this test.\n"
    "And you should run this test on an otherwise idle machine.\n";

double f1 = 0, f2 = 0;

BOOST_AUTO_TEST_CASE( continuous_use ) {
    BOOST_TEST_MESSAGE("Testing one long period." );
    siena_impl::Timer T;

    T.start();
    double start = read_current_utime();

    f1 = F(45);
    f2 = F(46);

    T.stop();
    double stop = read_current_utime();

    double t_r = stop - start;
    double t_t = T.read_microseconds();

    double e = percent_error(t_r, t_t);

    BOOST_REQUIRE_MESSAGE((e < 1), "t_t=" << t_t << ", t_r=" << t_r << ", err%=" << e 
			  << "\nexpecting err% < 1\n" << warning_msg);
}

BOOST_AUTO_TEST_CASE( intermittent_use ) {
    BOOST_TEST_MESSAGE("Testing two long periods." );
    siena_impl::Timer T;


    T.start();
    double start = read_current_utime();
    f1 = F(47);
    T.stop();
    f2 = F2(47);
    double stop = read_current_utime();

    double t_r = stop - start;
    double t_t = T.read_microseconds() * 2;

    double e = percent_error(t_r, t_t);

    BOOST_REQUIRE_MESSAGE((e < 1), "t_t=" << t_t << ", t_r=" << t_r << ", err%=" << e
			  << "\nexpecting err% < 1\n" << warning_msg);
}

BOOST_AUTO_TEST_CASE( frequent_use ) {
    BOOST_TEST_MESSAGE("Testing several short periods." );
    siena_impl::Timer T;

    for(int i = 0; i < 1000; ++i) {
	T.start();
	f1 += F((i % 100)/10 + 15);
	T.stop();
    }

    double start = read_current_utime();
    for(int i = 0; i < 1000; ++i) {
	f1 += F((i % 100)/10 + 15);
    }
    double stop = read_current_utime();

    double t_r = stop - start;
    double t_t = T.read_microseconds();

    double e = percent_error(t_r, t_t);

    BOOST_REQUIRE_MESSAGE((e < 3), "t_t=" << t_t << ", t_r=" << t_r << ", err%=" << e
			  << "\nexpecting err% < 3\n" << warning_msg);
}

#else // no timers 

BOOST_AUTO_TEST_CASE( void_test ) {
    BOOST_MESSAGE( "Timer feature is disabled." );
}

#endif


