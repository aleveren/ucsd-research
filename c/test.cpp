#include <iostream>
#include <cmath>
#include <vector>
#include <boost/random.hpp>

using namespace std;

const int ROWS = 100;
const int COLS = 4;

int main(int argc, char** argv) {
  long seed = 1;
  boost::random::mt19937 gen(seed);

  boost::random::uniform_real_distribution<> unif_distrib(0.0, 1.0);
  boost::random::normal_distribution<> norm_distrib(0.0, 1.0);
  boost::random::uniform_on_sphere<> sphere_distrib(COLS);
  boost::random::uniform_int_distribution<> int_distrib(0, COLS-1);

  int random_row_index =
      boost::random::uniform_int_distribution<>(0, ROWS-1)(gen);

  for (int i = 0; i < ROWS; i++) {
    double u = unif_distrib(gen);
    double v = norm_distrib(gen);
    cout << "u = " << u << ", v = " << v << ", w = ";

    vector<double> w = sphere_distrib(gen);
    int randomIndex = int_distrib(gen);
    w[randomIndex] = sqrt(COLS) + 1.0;
    if (i == random_row_index) {
      for (int j = 0; j < COLS; j++) {
        w[j] = 1.0;
      }
    }

    int j = 0;
    double sumSq = 0.0;
    for (double value: w) {
      if (j > 0) {
        cout << ", ";
      }
      cout << value;
      sumSq += value * value;
      j++;
    }
    double magnitude = sqrt(sumSq);
    cout << " (magnitude[w] = " << magnitude << ")" << endl;
  }

  return 0;
}
