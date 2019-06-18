#ifndef ACTIVATION_LAYER_HPP
#define ACTIVATION_LAYER_HPP

#include <functional>
#include "Layer.hpp"

/**
 * @brief Activation function as it's own layer
 */
class ActivationLayer : public AbstractBaseLayer {
 public:
  ActivationLayer(std::function<double(double)> activationFunction_,
                  std::function<double(double)> activationFunctionPrime_) {
    activationFunction = activationFunction_;
    activationFunctionPrime = activationFunctionPrime_;
  }
  /**
   * @copydoc AbstractBaseLayer::forwardPropagation
   */
  Eigen::VectorXd forwardPropagation(Eigen::VectorXd input_data) {
    input = input_data;
    output = input.unaryExpr(activationFunction);
    return output;
  }
  /**
   * @copydoc AbstractBaseLayer::backwardPropagation
   */
  Eigen::VectorXd backwardPropagation(Eigen::VectorXd output_error,
                                      double learningRate) {
    return input.unaryExpr(activationFunctionPrime).cwiseProduct(output_error);
  }
  /**
   * @copydoc AbstractBaseLayer::saveConfiguration
   * This does nothing as nothing has to be saved
   */
  void saveConfiguration(std::string path) const {}
  /**
   * @copydoc AbstractBaseLayer::loadConfiguration
   * This does nothing as nothing has to be loaded
   */
  void loadConfiguration(std::string path) {}

 private:
  std::function<double(double)> activationFunction; ///< activation function
  std::function<double(double)> activationFunctionPrime; ///< derivative of activation function
};

#endif