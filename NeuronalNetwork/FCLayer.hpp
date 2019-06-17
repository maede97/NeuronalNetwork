#ifndef FCLAYER_HPP
#define FCLAYER_HPP

#include "Layer.hpp"

/**
 * @brief Fully Connected Layer
 */
class FCLayer : public AbstractBaseLayer {
 public:
  /**
   * @brief Create FCLayer
   * @param input_size How many inputs there are
   * @param output_size How many outputs there are
   */
  FCLayer(unsigned int input_size, unsigned int output_size) {
    weights = Eigen::MatrixXd::Random(input_size, output_size);
    bias = Eigen::MatrixXd::Random(output_size, 1);
  }
  /**
   * @copydoc AbstractBaseLayer::forwardPropagation
   */
  Eigen::VectorXd forwardPropagation(Eigen::VectorXd input_data) {
    input = input_data;
    output = weights.transpose() * input + bias;
    return output;
  }
  /**
   * @copydoc AbstractBaseLayer::backwardPropagation
   */
  Eigen::VectorXd backwardPropagation(Eigen::VectorXd output_error,
                                      double learningRate) {
    Eigen::VectorXd input_error =
        output_error.transpose() * weights.transpose();
    Eigen::MatrixXd weights_error = input * output_error.transpose();

    weights -= learningRate * weights_error;
    bias -= learningRate * output_error;
    return input_error;
  }

 private:
  Eigen::MatrixXd weights; ///< internal weights
  Eigen::VectorXd bias; ///< internal bias
};

#endif