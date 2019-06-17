# NeuronalNetwork
[![Build Status](https://travis-ci.com/maede97/NeuronalNetwork.svg?branch=master)](https://travis-ci.com/maede97/NeuronalNetwork)

This repository contains an example C++ supervised neuronal network.

# Installation
## Requirements
* Eigen3
* Cmake
* Make
* C++
* Doxygen (for Documentation)

To install, simply run `./make.sh` in the root folder of this repository. This will also automatically run the executable with a simple XOR-Test.

# Example
Use following lines as a simple example.
```cpp
#include <cassert>
#include <cmath>

#include "NeuronalNetwork/includes.hpp"

int main() {
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

  Eigen::MatrixXd x_train(4, 2);
  Eigen::MatrixXd y_train(4, 1);
  x_train << 0, 0, 0, 1, 1, 0, 1, 1;
  y_train << 0, 1, 1, 0;

  Network net = Network();

  // Create all Layers
  FCLayer first(2, 3); // 2 input, 3 hidden
  ActivationLayer activ1(activation, activationPrime);
  FCLayer second(3, 1); // 3 hidden, 1 output
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
    std::cout << "Test:\t" << out.row(i).transpose() << "\t"
        << y_train.row(i) << std::endl;
  }

  return 0;
}
```

# Further information
This project was highly inspired by https://medium.com/datadriveninvestor/math-neural-network-from-scratch-in-python-d6da9f29ce65.