# NeuronalNetwork
[![Build Status](https://travis-ci.com/maede97/NeuronalNetwork.svg?branch=master)](https://travis-ci.com/maede97/NeuronalNetwork)

[![Documentation](https://codedocs.xyz/maede97/SudokuSolver.svg)](https://maede97.github.io/NeuronalNetwork)

This repository contains an example C++ supervised neuronal network.

# Installation
## Requirements
* Eigen3
* CMake
* Make
* C++

To install, simply run `./make.sh` in the root folder of this repository. To run the tests, execute `./test.sh`.

# Example
Please have a look at the [example](https://github.com/maede97/NeuronalNetwork/blob/master/examples/Example.cpp) or just copy this code here:
```cpp
#include <NeuronalNetwork/All>
#include <cmath>  // for tanh

int main() {
  // create activation and derivative
  auto activation = [](double x) -> double { return tanh(x); };
  auto activationPrime = [](double x) -> double {
    return 1 - tanh(x) * tanh(x);
  };

  // create loss and derivative
  auto loss = [](Eigen::VectorXd y_true, Eigen::VectorXd y_pred) -> double {
    return (y_pred - y_true).squaredNorm() / ((double)y_true.size());
  };
  auto loss_prime = [](Eigen::VectorXd y_true,
                       Eigen::VectorXd y_pred) -> Eigen::VectorXd {
    return 2.0 / ((double)y_true.size()) * (y_pred - y_true);
  };

  // create data
  DataSet data(2,1);
  data.addTrainingSet(Eigen::Vector2d(0, 0), Eigen::VectorXd::Zero(1));
  data.addTrainingSet(Eigen::Vector2d(0, 1), Eigen::VectorXd::Ones(1));
  data.addTrainingSet(Eigen::Vector2d(1, 0), Eigen::VectorXd::Ones(1));
  data.addTrainingSet(Eigen::Vector2d(1, 1), Eigen::VectorXd::Zero(1));

  data.addTestSet(Eigen::Vector2d(0, 0));
  data.addTestSet(Eigen::Vector2d(0, 1));
  data.addTestSet(Eigen::Vector2d(1, 0));
  data.addTestSet(Eigen::Vector2d(1, 1));

  // create Network
  Network net = Network();
  net.setData(data);

  // Create all Layers
  FCLayer first(2, 3);  // 2 input, 3 hidden
  ActivationLayer activ1(3, activation, activationPrime);
  FCLayer second(3, 1);  // 3 hidden, 1 output
  ActivationLayer activ2(1, activation, activationPrime);

  // add Layers in order
  net.add(&first);
  net.add(&activ1);
  net.add(&second);
  net.add(&activ2);

  // specify loss function
  net.use(loss, loss_prime);

  // train network
  net.fit(1000, 0.1);

  // get final output
  Eigen::MatrixXd out = net.predict();
  for (unsigned int i = 0; i < out.rows(); i++) {
    std::cout << "Test:\t" << out.row(i).transpose() << "\t"
              << data.getOutputTrainingData().row(i) << std::endl;
  }

  return 0;
}
```

# Documentation
This code is documented using Doxygen. Because of travis, this is not automatically done. The Doxygen configuration can be found [here](https://github.com/maede97/NeuronalNetwork/blob/master/doc/Doxyfile.in).

# Further information / Source
 - https://medium.com/datadriveninvestor/math-neural-network-from-scratch-in-python-d6da9f29ce65
 - http://yann.lecun.com/exdb/mnist/