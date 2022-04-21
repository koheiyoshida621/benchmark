#include <iostream>
#include <cstdio> // for perror
#include <ctime>
#include <string>  
#include <unistd.h>


using namespace std;

void getTime(string msg) {

  struct timespec ts;
  struct tm t;
  int ret;
  char hostname[16];

  // Get epoch time
  ret = clock_gettime(CLOCK_REALTIME, &ts);
  if (ret < 0) {
    perror("clock_gettime fail");
  }

  // Convert into local and parsed time
  localtime_r(&ts.tv_sec, &t);

  // Create string with strftime
  char buf[32];
  ret = strftime(buf, 32, "%Y/%m/%d %H:%M:%S", &t);
  if (ret == 0) {
    perror("strftime fail");
  }

  // Add milli-seconds with snprintf
  char output[32];
  const int msec = ts.tv_nsec / 1000000;
  ret = snprintf(output, 32, "%s.%03d", buf, msec);
  if (ret == 0) {
    perror("snprintf fail");
  }
    
  // Get Hostname
  gethostname(hostname, sizeof(hostname));

  // Result
  std::cout << '\n'  << std::string(output) << ',' << std::string(hostname) << ','  <<  std::string(msg) << std::endl;

}

// for test
/*
int main() {
  char passbuf[] = "/gs/hs0/tgh-21IAH/yoshida/Trash/test/getTime";
getTime("trst start");
}
*/
