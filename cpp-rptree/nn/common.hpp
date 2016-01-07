#include <vector>
#include <string>
#include <boost/random.hpp>
#include <ctime>

/* Linear algebra */

double dot(std::vector<double> a, std::vector<double> b);

double euclidean(std::vector<double> a, std::vector<double> b);

double magnitude(std::vector<double> a);

std::vector<double> randomUnitVector(int dim);

typedef std::vector<std::vector<double> *> Data;

typedef double (*DistanceMetric)(std::vector<double>, std::vector<double>);

std::vector<double> * linearScanNearestNeighbor(
    Data data, std::vector<double> query, DistanceMetric metric);

Data loadData(std::string path, int columnsToIgnore);

extern boost::random::mt19937 gen;

/* Quantiles */

double selectRank(std::vector<double> values, int rank);

double selectQuantile(std::vector<double> values, double alpha);

/* Strings */

std::string join(std::string sep, std::vector<double> vec);

/* Time */

#define TIMER_START(name) \
  clock_t _MACRO_startTime##name = clock(); \
  cout << "Starting timer [" #name "]" << endl;

#define TIMER_END(name) \
  double _MACRO_elapsedSeconds##name = \
      (double) (clock() - _MACRO_startTime##name) / CLOCKS_PER_SEC; \
  cout << "Elapsed seconds [" #name "] = " \
      << _MACRO_elapsedSeconds##name << endl;
