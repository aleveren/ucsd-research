#include "utils.hpp"

#include <iostream>
#include <vector>
#include <exception>
#include <boost/random.hpp>

using namespace std;

int main(int argc, char** argv) {
  string path = "../data/testdata.csv";
  int columnsToIgnore = 1;
  if (argc > 1) {
    string analysis = string(argv[1]);
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

  cout << "Loading data from " << path << " ... " << flush;
  Data data = loadData(path, columnsToIgnore);

  int COLS = data[0]->size();
  int ROWS = data.size();

  cout << "Data size: COLS = " << COLS << ", ROWS = " << ROWS << endl;

  int maxLeafSize = 500;
  int numTrees = 10;

  cout << "Building trees" << endl;
  Forest *forest = makeForest(data, maxLeafSize, numTrees, &euclidean);
  vector<double> query(COLS, 0.0);

  cout << "Running random-projection query ... " << flush;
  vector<double> *result = forest->nearestNeighbor(query);
  double resultDistance = euclidean(*result, query);
  cout << "Found point at distance " << resultDistance << endl;

  cout << "Running linear scan ... " << flush;
  result = linearScanNearestNeighbor(data, query, &euclidean);
  resultDistance = euclidean(*result, query);
  cout << "Found point at distance " << resultDistance << endl;

  return 0;
}
