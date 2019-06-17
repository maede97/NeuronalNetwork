/**
@page getting_started Getting Started
```
#include <cmath> // for tanh

#include "NeuronalNetwork/includes.hpp"

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
  Eigen::MatrixXd x_train(4, 2);
  Eigen::MatrixXd y_train(4, 1);
  x_train << 0, 0, 0, 1, 1, 0, 1, 1;
  y_train << 0, 1, 1, 0;

  // create Network
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

  // specify loss function
  net.use(loss, loss_prime);

  // train network
  net.fit(x_train, y_train, 1000, 0.1);

  // get final output
  Eigen::MatrixXd out = net.predict(x_train);
  for (unsigned int i = 0; i < out.rows(); i++) {
    std::cout << "Test:\t" << out.row(i).transpose() << "\t"
        << y_train.row(i) << std::endl;
  }

  return 0;
}
```
*/