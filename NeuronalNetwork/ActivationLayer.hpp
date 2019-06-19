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
   * @param activationFunctionPrime The derivative of the function
   */
  ActivationLayer(const unsigned int input_size,
                  std::function<double(double)> activationFunction,
                  std::function<double(double)> activationFunctionPrime) {
    activationFunction_ = activationFunction;
    activationFunctionPrime_ = activationFunctionPrime;
    input_ = Eigen::VectorXd(input_size);
  }
  /**
   * @copydoc AbstractBaseLayer::forwardPropagation
   */
  Eigen::VectorXd forwardPropagation(const Eigen::VectorXd& input_data) {
    input_ = input_data;
    return input_.unaryExpr(activationFunction_);
  }
  /**
   * @copydoc AbstractBaseLayer::backwardPropagation
   */
  Eigen::VectorXd backwardPropagation(const Eigen::VectorXd& output_error,
                                      const double learningRate) {
    return input_.unaryExpr(activationFunctionPrime_)
        .cwiseProduct(output_error);
  }
  /**
   * @copydoc AbstractBaseLayer::saveConfiguration
   * @note This does nothing as nothing has to be saved.
   */
  void saveConfiguration(const std::string& path) const {}
  /**
   * @copydoc AbstractBaseLayer::loadConfiguration
   * @note This does nothing as nothing has to be loaded.
   */
  void loadConfiguration(const std::string& path) {}

 private:
  std::function<double(double)> activationFunction_;  ///< activation function
  std::function<double(double)>
      activationFunctionPrime_;  ///< derivative of activation function
};

#endif