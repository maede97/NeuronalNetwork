#ifndef FCLAYER_HPP
#define FCLAYER_HPP

#include "Layer.hpp"

class FCLayer : public AbstractBaseLayer {
 public:
  FCLayer(unsigned int input_size, unsigned int output_size) {
    weights = Eigen::MatrixXd::Random(input_size, output_size);
    bias = Eigen::MatrixXd::Random(output_size, 1);
  }

  Eigen::VectorXd forwardPropagation(Eigen::VectorXd input_data) {
    input = input_data;
    output = weights.transpose() * input + bias;
    return output;
  }

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
  Eigen::MatrixXd weights;
  Eigen::VectorXd bias;
};

#endif