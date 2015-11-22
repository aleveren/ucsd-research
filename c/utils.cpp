#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <exception>
#include <boost/random.hpp>

using namespace std;

long seed = 1;
boost::random::mt19937 gen(seed);

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

typedef vector<vector<double> *> Data;
typedef double (*DistanceMetric)(vector<double>, vector<double>);

DistanceMetric defaultMetric = &euclidean;

class Tree {
public:
  virtual vector<double> * nearestNeighbor(vector<double>) = 0;
  virtual Tree * getLeaf(vector<double>) = 0;
};

class Forest {
public:
  Forest(vector<Tree*> trees, DistanceMetric metric) :
      trees_(trees), metric_(metric) { }

  vector<double> * nearestNeighbor(vector<double> query) {
    vector<double> *nearest = nullptr;
    double nearestDistance = 0.0;
    for (int i = 0; i < trees_.size(); i++) {
      vector<double> *currentNearest = trees_[i]->nearestNeighbor(query);
      double currentDistance = metric_(query, *currentNearest);
      if (nearest == nullptr || currentDistance < nearestDistance) {
        nearest = currentNearest;
        nearestDistance = currentDistance;
      }
    }
    if (nearest == nullptr) {
      throw runtime_error("Internal error: no nearest neighbor");
    }
    return nearest;
  }

private:
  vector<Tree*> trees_;
  DistanceMetric metric_;
};

class Rule {
public:
  Rule(vector<double> direction, double threshold, double approxQuantile) :
      direction_(direction),
      threshold_(threshold),
      approxQuantile_(approxQuantile) { }

  bool operator()(vector<double> row) {
    return dot(direction_, row) <= threshold_;
  }

private:
  //friend std::ostream& operator<<(std::ostream&, const Rule&); 
  vector<double> direction_;
  double threshold_;
  double approxQuantile_;
};

//std::ostream& operator<<(std::ostream& s, const Rule& rule) {
//  return s << "Rule("
//      << "direction: " << rule.direction_ << ", "
//      << "threshold: " << rule.threshold_ << ", "
//      << "approxQuantile: " << rule.approxQuantile_ << ")";
//}

class Node : public Tree {
public:
  Node(Rule rule, Tree *leftTree, Tree *rightTree) :
      rule_(rule),
      leftTree_(leftTree),
      rightTree_(rightTree) { }

  vector<double> * nearestNeighbor(vector<double> query) {
    return getLeaf(query)->nearestNeighbor(query);
  }

  Tree * getLeaf(vector<double> query) {
    return rule_(query) ? leftTree_ : rightTree_;
  }

private:
  Rule rule_;
  Tree *leftTree_;
  Tree *rightTree_;
};

class Leaf : public Tree {
public:
  Leaf(Data data, DistanceMetric metric) :
      data_(data), metric_(metric) { }

  vector<double> * nearestNeighbor(vector<double> query) {
    vector<double> *nearest = nullptr;
    double nearestDistance = 0.0;
    for (int i = 0; i < data_.size(); i++) {
      double currentDistance = metric_(query, *data_[i]);
      if (nearest == nullptr || currentDistance < nearestDistance) {
        nearestDistance = currentDistance;
        nearest = data_[i];
      }
    }
    if (nearest == nullptr) {
      throw runtime_error("Internal error: no nearest neighbor");
    }
    return nearest;
  }

  Tree * getLeaf(vector<double> query) {
    return this;
  }

private:
  Data data_;
  DistanceMetric metric_;
};

Rule chooseRule(Data data) {
  int dim = data[0]->size();
  vector<double> direction = randomUnitVector(dim);

  vector<double> projections;
  for (int i = 0; i < data.size(); i++) {
    double currentProjection = dot(*data[i], direction);
    projections.push_back(currentProjection);
  }

  double randomQuantile =
      boost::random::uniform_real_distribution<>(0.25, 0.75)(gen);

  double threshold = selectQuantile(projections, randomQuantile);
  return Rule(direction, threshold, randomQuantile);
}

Tree * makeTree(Data data, int maxLeafSize, DistanceMetric metric) {
  if (data.size() <= maxLeafSize) {
    return new Leaf(data, metric);
  }
  Rule rule = chooseRule(data);

  Data leftData;
  Data rightData;

  // Partition the data
  for (int i = 0; i < data.size(); i++) {
    vector<double> *row = data[i];
    if (rule(*row)) {
      leftData.push_back(row);
    } else {
      rightData.push_back(row);
    }
  }

  Tree *leftTree = makeTree(leftData, maxLeafSize, metric);
  Tree *rightTree = makeTree(rightData, maxLeafSize, metric);
  return new Node(rule, leftTree, rightTree);
}

Forest makeForest(
    Data data, int maxLeafSize, int numTrees, DistanceMetric metric) {
  vector<Tree*> trees;
  for (int i = 0; i < numTrees; i++) {
    Tree *tree = makeTree(data, maxLeafSize, metric);
    trees.push_back(tree);
  }
  return Forest(trees, metric);
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

int main(int argc, char** argv) {
  // TODO: keep this in a separate file, so that the rest can be
  // used as a library
  
  const int ROWS = 500;
  const int COLS = 5;

  boost::random::uniform_real_distribution<> unif_distrib(0.0, 1.0);
  boost::random::uniform_int_distribution<> column_distrib(0, COLS-1);
 
  int random_row_index =
      boost::random::uniform_int_distribution<>(0, ROWS-1)(gen);

  Data data;

  double M = sqrt(COLS) + 1.0;
  cout << "Building dataset" << endl;
  for (int i = 0; i < ROWS; i++) {
    vector<double> *w = new vector<double>(COLS, 0.0);

    if (i == random_row_index) {
      for (int j = 0; j < COLS; j++) {
        (*w)[j] = 1.0;
      }
    } else {
      int random_col_index = column_distrib(gen);
      for (int j = 0; j < COLS; j++) {
        if (j == random_col_index) {
          (*w)[j] = M;
        } else {
          (*w)[j] = unif_distrib(gen);
        }
      }
    }

    data.push_back(w);
  }

  cout << "Building trees" << endl;
  Forest forest = makeForest(data, /* maxLeafSize */ 100,
      /* numTrees */ 20, /* metric */ &euclidean);
  vector<double> query(COLS, 0.0);

  cout << "Running random-projection query ... ";
  vector<double> *result = forest.nearestNeighbor(query);
  double resultDistance = euclidean(*result, query);
  cout << "Found point at distance " << resultDistance << endl;

  cout << "Running linear scan ... ";
  result = linearScanNearestNeighbor(data, query, &euclidean);
  resultDistance = euclidean(*result, query);
  cout << "Found point at distance " << resultDistance << endl;

  return 0;
}
