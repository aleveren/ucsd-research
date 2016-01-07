#include "../common.hpp"
#include <vector>

class Rule {
public:
  Rule(std::vector<double> direction, double threshold, double approxQuantile) :
      direction_(direction),
      threshold_(threshold),
      approxQuantile_(approxQuantile) { }

  bool operator()(std::vector<double> row) {
    return dot(direction_, row) <= threshold_;
  }

  static Rule chooseRule(Data data);

private:
  std::vector<double> direction_;
  double threshold_;
  double approxQuantile_;
};

class Tree {
public:
  virtual std::vector<double> * nearestNeighbor(std::vector<double>) = 0;
  virtual Tree * getLeaf(std::vector<double>) = 0;

  static Tree * makeTree(Data data, int maxLeafSize, DistanceMetric metric);
};

class Leaf : public Tree {
public:
  Leaf(Data data, DistanceMetric metric) :
      data_(data), metric_(metric) { }

  std::vector<double> * nearestNeighbor(std::vector<double> query);

  Tree * getLeaf(std::vector<double> query) {
    return this;
  }

private:
  Data data_;
  DistanceMetric metric_;
};

class Node : public Tree {
public:
  Node(Rule rule, Tree *leftTree, Tree *rightTree) :
      rule_(rule),
      leftTree_(leftTree),
      rightTree_(rightTree) { }

  std::vector<double> * nearestNeighbor(std::vector<double> query) {
    return getLeaf(query)->nearestNeighbor(query);
  }

  Tree * getLeaf(std::vector<double> query) {
    return rule_(query) ? leftTree_ : rightTree_;
  }

private:
  Rule rule_;
  Tree *leftTree_;
  Tree *rightTree_;
};

class Forest {
public:
  Forest(std::vector<Tree*> trees, DistanceMetric metric) :
      trees_(trees), metric_(metric) { }

  std::vector<double> * nearestNeighbor(std::vector<double> query);

  static Forest * makeForest(
      Data data, int maxLeafSize, int numTrees, DistanceMetric metric);

private:
  std::vector<Tree*> trees_;
  DistanceMetric metric_;
};

