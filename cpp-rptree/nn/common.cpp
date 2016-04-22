#include "common.hpp"
#include <boost/random.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <algorithm>
#include <sstream>

using namespace std;

/* Linear algebra */

long seed = 1;
boost::random::mt19937 gen(seed);

double dot(vector<double> a, vector<double> b) {
  double result = 0.0;
  for (int i = 0; i < a.size(); i++) {
    result += a[i] * b[i];
  }
  return result;
}

double euclidean(vector<double> a, vector<double> b) {
  double sumSq = 0.0;
  for (int i = 0; i < a.size(); i++) {
    double diff = a[i] - b[i];
    sumSq += diff * diff;
  }
  return sqrt(sumSq);
}

double magnitude(vector<double> a) {
  double sumSq = 0.0;
  for (int i = 0; i < a.size(); i++) {
    double value = a[i];
    sumSq += value * value;
  }
  return sqrt(sumSq);
}

vector<double> randomUnitVector(int dim) {
  boost::random::uniform_on_sphere<> distrib(dim);
  return distrib(gen);
}

vector<double> * linearScanNearestNeighbor(
    Data data, vector<double> query, DistanceMetric metric) {
  vector<double> *nearest = nullptr;
  double nearestDistance = 0.0;
  for (int i = 0; i < data.size(); i++) {
    double currentDistance = metric(query, *data[i]);
    if (nearest == nullptr || currentDistance < nearestDistance) {
      nearestDistance = currentDistance;
      nearest = data[i];
    }
  }
  if (nearest == nullptr) {
    throw runtime_error("Internal error: no nearest neighbor");
  }
  return nearest;
}

Data loadData(string path, int columnsToIgnore) {
  Data result;
  string line;
  ifstream f(path);
  if (!f.is_open()) {
    throw runtime_error("Could not open file");
  }
  getline(f, line);  // Discard the header line
  while (getline(f, line)) {
    vector<string> columns;
    boost::split(columns, line, boost::is_any_of(","));
    vector<double> *dColumns = new vector<double>();
    int colIndex = 0;
    for (string value : columns) {
      // TODO: make this more general / parameterized
      if (colIndex >= columnsToIgnore) {
        double dValue;
        try {
          dValue = boost::lexical_cast<double>(value);
        } catch (boost::bad_lexical_cast e) {
          throw runtime_error("Could not parse CSV value as double");
        }
        dColumns->push_back(dValue);
      }
      colIndex++;
    }
    result.push_back(dColumns);
  }
  f.close();
  return result;
}

/* Quantiles */

double selectRank(vector<double> values, int rank) {
  std::nth_element(values.begin(), values.begin() + rank, values.end());
  return values[rank];
}

double selectQuantile(vector<double> values, double alpha) {
  int N = values.size();
  int unboundedRank = (int) (N * alpha);
  int rank = min(N, max(0, unboundedRank));
  return selectRank(values, rank);
}

/* Strings */

string join(string sep, vector<double> vec) {
  stringstream ss;
  int i = 0;
  for (auto value: vec) {
    if (i > 0) {
      ss << sep;
    }
    ss << value;
    i++;
  }
  return ss.str();
}
