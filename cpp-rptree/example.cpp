#include "utils.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <exception>
#include <boost/random.hpp>

using namespace std;

double earthmover(vector<double> a, vector<double> b) {
  // TODO: double-check this implementation via testing
  double aTotal = 0.0;
  double bTotal = 0.0;
  for (int i = 0; i < a.size(); i++) {
    aTotal += max(0.0, a[i]);
    bTotal += max(0.0, b[i]);
  }
  if (aTotal <= 0 || bTotal <= 0) {
    throw runtime_error("Cannot compute earthmover distance");
  }
  double aSum = 0.0;
  double bSum = 0.0;
  double result = 0.0;
  for (int i = 0; i < a.size(); i++) {
    double aScaled = max(0.0, a[i]) / aTotal;
    double bScaled = max(0.0, b[i]) / bTotal;
    aSum += aScaled;
    bSum += bScaled;
    result += abs(aSum - bSum);
  }
  return result;
}

double normalizedDotProduct(vector<double> a, vector<double> b) {
  // TODO: double-check this implementation via testing
  double numer = abs(dot(a, b));  // TODO: how to handle negative components?
  double denom = magnitude(a) * magnitude(b);
  if (denom <= 0) {
    throw runtime_error("Cannot compute normalized dot product");
  }
  return 1.0 - numer / denom;
}

int main(int argc, char** argv) {
  TIMER_START(overall);

  string path = "../data/testdata.csv";
  int columnsToIgnore = 1;
  string analysis = "sim";

  if (argc > 1) {
    analysis = string(argv[1]);
    if (analysis == "sim") {
      // Do nothing; this is the default.
    } else if (analysis == "full") {
      path = "../data/accumDataRDR_all.csv";
      columnsToIgnore = 3;
    } else if (analysis == "subset") {
      path = "../data/accumDataRDR_subset.csv";
      columnsToIgnore = 3;
    } else {
      throw runtime_error(string("Unrecognized analysis: ") + analysis);
    }
  }

  DistanceMetric metric = &euclidean;
  string metricName = "euclidean";
  if (argc > 2) {
    metricName = string(argv[2]);
    if (metricName == "euclidean") {
      // Do nothing; this is the default.
    } else if (metricName == "normalizedDotProduct") {
      metric = &normalizedDotProduct;
    } else if (metricName == "earthmover") {
      metric = &earthmover;
    } else {
      throw runtime_error(string("Unrecognized metricName: ") + metricName);
    }
  }

  Data data;
  int COLS;
  int ROWS;
  TIMER_START(loadData);
  cout << "Loading data from " << path << " ... " << flush;
  data = loadData(path, columnsToIgnore);
  COLS = data[0]->size();
  ROWS = data.size();
  cout << "Data size: COLS = " << COLS << ", ROWS = " << ROWS << endl;
  TIMER_END(loadData);

  vector<double> query(COLS, 0.0);
  if (analysis == "subset" || analysis == "full") {
    boost::random::mt19937 gen(1);
    for (int i = 0; i < COLS; i++) {
      query[i] = boost::random::uniform_real_distribution<>(0.0, 10.0)(gen);
    }
  }
  cout << "Query vector: " << join(", ", query) << endl;

  int maxLeafSize = 500;
  int numTrees = 10;

  Forest *forest;
  TIMER_START(buildingTrees);
  cout << "Building trees" << endl;
  forest = makeForest(data, maxLeafSize, numTrees, metric);
  TIMER_END(buildingTrees);

  TIMER_START(rpQuery);
  cout << "Running random-projection query ... " << flush;
  vector<double> *result = forest->nearestNeighbor(query);
  double resultDistance = metric(*result, query);
  cout << "Found point at distance " << resultDistance << endl;
  TIMER_END(rpQuery);

  TIMER_START(linearScanQuery);
  cout << "Running linear scan ... " << flush;
  vector<double> *result = linearScanNearestNeighbor(data, query, metric);
  double resultDistance = metric(*result, query);
  cout << "Found point at distance " << resultDistance << endl;
  TIMER_END(linearScanQuery);

  TIMER_END(overall);

  return 0;
}
