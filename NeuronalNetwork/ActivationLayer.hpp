#ifndef ACTIVATION_LAYER_HPP
#define ACTIVATION_LAYER_HPP

#include <functional>
#include "Layer.hpp"

class ActivationLayer : public AbstractBaseLayer {
 public:
  ActivationLayer(std::function<double(double)> activationFunction_,
                  std::function<double(double)> activationFunctionPrime_) {
    activationFunction = activationFunction_;
    activationFunctionPrime = activationFunctionPrime_;
  }

  Eigen::VectorXd forwardPropagation(Eigen::VectorXd input_data) {
    input = input_data;
    output = input.unaryExpr(activationFunction);
    return output;
  }

  Eigen::VectorXd backwardPropagation(Eigen::VectorXd output_error,
                                      double learningRate) {
    return input.unaryExpr(activationFunctionPrime).cwiseProduct(output_error);
  }

 private:
  std::function<double(double)> activationFunction;
  std::function<double(double)> activationFunctionPrime;
};

#endif