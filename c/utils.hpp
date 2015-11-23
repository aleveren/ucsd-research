#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <exception>
#include <boost/random.hpp>
#include <boost/algorithm/string.hpp>

double selectRank(std::vector<double> values, int rank);
double selectQuantile(std::vector<double> values, double alpha);
double dot(std::vector<double> a, std::vector<double> b);
double euclidean(std::vector<double> a, std::vector<double> b);
double magnitude(std::vector<double> a);
std::vector<double> randomUnitVector(int dim);
std::string join(std::string sep, std::vector<double> vec);

typedef std::vector<std::vector<double> *> Data;
typedef double (*DistanceMetric)(std::vector<double>, std::vector<double>);

class Tree;
class Rule;

class Forest {
public:
  Forest(std::vector<Tree*> trees, DistanceMetric metric);
  std::vector<double> * nearestNeighbor(std::vector<double> query);

private:
  std::vector<Tree*> trees_;
  DistanceMetric metric_;
};

Rule chooseRule(Data data);
Tree * makeTree(Data data, int maxLeafSize, DistanceMetric metric);
Forest * makeForest(
    Data data, int maxLeafSize, int numTrees, DistanceMetric metric);
std::vector<double> * linearScanNearestNeighbor(
    Data data, std::vector<double> query, DistanceMetric metric);
Data loadData(std::string path, int columnsToIgnore);

