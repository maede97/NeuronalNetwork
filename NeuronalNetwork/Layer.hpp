#ifndef LAYER_HPP
#define LAYER_HPP

#include <Eigen/Dense>

class AbstractBaseLayer {
 public:
  virtual Eigen::VectorXd forwardPropagation(Eigen::VectorXd input) = 0;
  virtual Eigen::VectorXd backwardPropagation(Eigen::VectorXd output,
                                              double learningRate) = 0;

 protected:
  Eigen::VectorXd input;
  Eigen::VectorXd output;
};
#endif