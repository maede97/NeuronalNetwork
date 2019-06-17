#ifndef LAYER_HPP
#define LAYER_HPP

#include <Eigen/Dense>

/**
 * @brief Abstract Layer class
 */
class AbstractBaseLayer {
 public:
  /**
   * @brief Propagate input through this layer
   * @param input_data Vector containing values to propagate
   * @return Vector containing propagated values
   */
  virtual Eigen::VectorXd forwardPropagation(Eigen::VectorXd input_data) = 0;
  /**
   * @brief Propagate output back through this layer
   * @param output_error Vector containing values to backpropagate
   * @param learningRate The rate for learning
   * @return Vector containing propagated values
   */
  virtual Eigen::VectorXd backwardPropagation(Eigen::VectorXd output_error,
                                              double learningRate) = 0;

 protected:
  Eigen::VectorXd input; ///< internal input of this layer
  Eigen::VectorXd output; ///< internal output of this layer
};
#endif