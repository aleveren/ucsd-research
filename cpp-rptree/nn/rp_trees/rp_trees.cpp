#include "rp_trees.hpp"
#include <stdexcept>
#include <iostream>
#include <vector>
#include <boost/random.hpp>

using namespace std;

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

Forest * Forest::makeForest(
    Data data, int maxLeafSize, int numTrees, DistanceMetric metric) {
  vector<Tree*> trees;
  for (int i = 0; i < numTrees; i++) {
    cout << "Building tree index " << i << endl;
    Tree *tree = Tree::makeTree(data, maxLeafSize, metric);
    trees.push_back(tree);
  }
  return new Forest(trees, metric);
}

Rule Rule::chooseRule(Data data) {
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

Tree * Tree::makeTree(Data data, int maxLeafSize, DistanceMetric metric) {
  if (data.size() <= maxLeafSize) {
    return new Leaf(data, metric);
  }
  Rule rule = Rule::chooseRule(data);

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

vector<double> * Leaf::nearestNeighbor(vector<double> query) {
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
