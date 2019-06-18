#ifndef ACTIVATION_LAYER_HPP
#define ACTIVATION_LAYER_HPP

#include <functional>
#include "AbstractBaseLayer.hpp"

/**
 * @brief Activation function as it's own layer
 */
class ActivationLayer : public AbstractBaseLayer {
 public:
  /**
   * @brief Construct an ActivationLayer
   * @param input_size Size for this Layer
   * @param activationFunction Use this function to activate (i.e. sigmoid)
   * @param activationFunctionPrime_ The derivative of the function
   */
  ActivationLayer(const unsigned int input_size, std::function<double(double)> activationFunction_,
                  std::function<double(double)> activationFunctionPrime_) {
    activationFunction = activationFunction_;
    activationFunctionPrime = activationFunctionPrime_;
    input = Eigen::VectorXd(input_size);
  }
  /**
   * @copydoc AbstractBaseLayer::forwardPropagation
   */
  Eigen::VectorXd forwardPropagation(Eigen::VectorXd input_data) {
    input = input_data;
    return input.unaryExpr(activationFunction);
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
   * @note This does nothing as nothing has to be saved.
   */
  void saveConfiguration(std::string path) const {}
  /**
   * @copydoc AbstractBaseLayer::loadConfiguration
   * @note This does nothing as nothing has to be loaded.
   */
  void loadConfiguration(std::string path) {}

 private:
  std::function<double(double)> activationFunction; ///< activation function
  std::function<double(double)> activationFunctionPrime; ///< derivative of activation function
};

#endif