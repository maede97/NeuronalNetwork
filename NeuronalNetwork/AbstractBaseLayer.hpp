#ifndef LAYER_HPP
#define LAYER_HPP

#include <eigen3/Eigen/Dense>
#include "DataSet.hpp"
#include <string>

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
  virtual Eigen::VectorXd forwardPropagation(const Eigen::VectorXd& input_data) = 0;
  /**
   * @brief Propagate output back through this layer and adjust itself
   * @param output_error Vector containing values to backpropagate
   * @param learningRate The rate for learning
   * @return Vector containing propagated values
   */
  virtual Eigen::VectorXd backwardPropagation(const Eigen::VectorXd& output_error,
                                              const double learningRate) = 0;
  /**
   * @brief Saves configuration of weights and bias to disk
   * @param path Where to write
   */
  virtual void saveConfiguration(const std::string& path) const = 0;

  /**
   * @brief Loads configuration of weights and bias from disk
   * @param path Where to write
   */
  virtual void loadConfiguration(const std::string& path) = 0;

 protected:
  Eigen::VectorXd input_;   ///< internal input of this layer
};
#endif