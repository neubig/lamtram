
#include <lamtram/timer.h>

#include <time.h>
#include <sys/time.h>
 
#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif
 
using namespace std;
using namespace lamtram;

void Timer::GetCurrentUTCTime(timespec &ts) {
 
#ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
    clock_serv_t cclock;
    mach_timespec_t mts;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    clock_get_time(cclock, &mts);
    mach_port_deallocate(mach_task_self(), cclock);
    ts.tv_sec = mts.tv_sec;
    ts.tv_nsec = mts.tv_nsec;
#else
    clock_gettime(CLOCK_REALTIME, &ts);
#endif
 
}

double Timer::Elapsed() {
    timespec end_time;
    Timer::GetCurrentUTCTime(end_time);
    return (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_nsec - start_time.tv_nsec) / 1000000000.0;
}
