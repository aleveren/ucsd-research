#include "utils.hpp"

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

DistanceMetric defaultMetric = &euclidean;

class Tree {
public:
  virtual vector<double> * nearestNeighbor(vector<double>) = 0;
  virtual Tree * getLeaf(vector<double>) = 0;
};

/* Beginning of Forest implementation */
Forest::Forest(vector<Tree*> trees, DistanceMetric metric) :
    trees_(trees), metric_(metric) { }

vector<double> * Forest::nearestNeighbor(vector<double> query) {
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
/* End of Forest implementation */

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

Forest * makeForest(
    Data data, int maxLeafSize, int numTrees, DistanceMetric metric) {
  vector<Tree*> trees;
  for (int i = 0; i < numTrees; i++) {
    cout << "Building tree index " << i << endl;
    Tree *tree = makeTree(data, maxLeafSize, metric);
    trees.push_back(tree);
  }
  return new Forest(trees, metric);
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

