#include <cassert>
#include "NeuronalNetwork/ActivationLayer.hpp"
#include "NeuronalNetwork/FCLayer.hpp"
#include "NeuronalNetwork/Layer.hpp"
#include "NeuronalNetwork/Network.hpp"

#include <cmath>

/*
 * Source: https://medium.com/datadriveninvestor/math-neural-network-from-scratch-in-python-d6da9f29ce65
 */

int clearOutput(double x) {
  if (x < 0.1) return 0;
  if (x > 0.9)
    return 1;
  else
    return -1;
}

void assertEqual(Eigen::VectorXd original, Eigen::VectorXd toTest) {
  for (int i = 0; i < toTest.size(); i++) {
    assert(clearOutput(toTest[i]) != -1);
    assert(clearOutput(toTest[i]) == (int)original[i]);
  }
}

void XOR(Eigen::MatrixXd& x_train, Eigen::MatrixXd& y_train) {
  x_train = Eigen::MatrixXd(4, 2);
  y_train = Eigen::MatrixXd(4, 1);
  x_train << 0, 0, 0, 1, 1, 0, 1, 1;
  y_train << 0, 1, 1, 0;
}
void AND(Eigen::MatrixXd& x_train, Eigen::MatrixXd& y_train) {
  x_train = Eigen::MatrixXd(4, 2);
  y_train = Eigen::MatrixXd(4, 1);
  x_train << 0, 0, 0, 1, 1, 0, 1, 1;
  y_train << 0, 0, 0, 1;
}
void OR(Eigen::MatrixXd& x_train, Eigen::MatrixXd& y_train) {
  x_train = Eigen::MatrixXd(4, 2);
  y_train = Eigen::MatrixXd(4, 1);
  x_train << 0, 0, 0, 1, 1, 0, 1, 1;
  y_train << 0, 1, 1, 1;
}

void NOT(Eigen::MatrixXd& x_train, Eigen::MatrixXd& y_train) {
  x_train = Eigen::MatrixXd(2, 1);
  y_train = Eigen::MatrixXd(2, 1);
  x_train << 0, 1;
  y_train << 1, 0;
}
int main() {
  srand(time(0));
  auto activation = [](double x) -> double { return tanh(x); };
  auto activationPrime = [](double x) -> double {
    return 1 - tanh(x) * tanh(x);
  };

  auto loss = [](Eigen::VectorXd y_true, Eigen::VectorXd y_pred) -> double {
    return (y_pred - y_true).squaredNorm() / ((double)y_true.size());
  };
  auto loss_prime = [](Eigen::VectorXd y_true,
                       Eigen::VectorXd y_pred) -> Eigen::VectorXd {
    return 2.0 / ((double)y_true.size()) * (y_pred - y_true);
  };

  Eigen::MatrixXd x_train;
  Eigen::MatrixXd y_train;
  XOR(x_train, y_train);

  Network net = Network();

  // Create all Layers
  FCLayer first(x_train.cols(), 3);
  ActivationLayer activ1(activation, activationPrime);
  FCLayer second(3, y_train.cols());
  ActivationLayer activ2(activation, activationPrime);

  // add Layers in order
  net.add(&first);
  net.add(&activ1);
  net.add(&second);
  net.add(&activ2);

  net.use(loss, loss_prime);

  net.fit(x_train, y_train, 1000, 0.1);

  Eigen::MatrixXd out = net.predict(x_train);
  for (unsigned int i = 0; i < out.rows(); i++) {
    std::cout << "Test:\t" << out.row(i).transpose() << "\tshould be equal to\t"
              << y_train.row(i) << std::endl;
    assertEqual(y_train.row(i), out.row(i));
  }

  return 0;
}