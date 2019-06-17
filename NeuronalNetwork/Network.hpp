#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <functional>
#include <iostream>
#include <vector>

#include "Layer.hpp"

class Network {
 public:
  Network() {}
  void add(AbstractBaseLayer *layer) { layers.push_back(layer); }
  void use(std::function<double(Eigen::VectorXd, Eigen::VectorXd)> loss_,
           std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd)>
               lossPrime_) {
    loss = loss_;
    lossPrime = lossPrime_;
  }

  // give each input data as a row in the input matrix
  std::vector<Eigen::VectorXd> predict(Eigen::MatrixXd input_data) {
    unsigned int samples = input_data.rows();
    std::vector<Eigen::VectorXd> result;
    Eigen::VectorXd output;
    for (unsigned int i = 0; i < samples; i++) {
      output = input_data.row(i).transpose();
      for (auto layer = layers.begin(); layer != layers.end(); layer++) {
        output = (*layer)->forwardPropagation(output);
      }
      result.push_back(output);
    }
    return result;
  }

  void fit(Eigen::MatrixXd x_train, Eigen::MatrixXd y_train,
           unsigned int epochs, double learning_rate) {
    unsigned int samples = x_train.rows();
    Eigen::VectorXd output;
    for (unsigned int i = 0; i < epochs; i++) {
      double err = 0;
      for (unsigned int j = 0; j < samples; j++) {
        output = x_train.row(j).transpose();
        for (auto layer = layers.begin(); layer != layers.end(); layer++) {
          output = (*layer)->forwardPropagation(output);
        }
        // compute for display purpose
        err += loss(y_train.row(j).transpose(), output);

        // backwards propagation
        Eigen::VectorXd error = lossPrime(y_train.row(j).transpose(), output);
        for (auto layer = layers.end() - 1; layer + 1 != layers.begin();
             layer--) {
          error = (*layer)->backwardPropagation(error, learning_rate);
        }
      }
      err /= (double)samples;
      std::cout << "Epoch: " << i + 1 << "\terror: " << err << "\r";
    }
    std::cout << std::endl;
  }

 private:
  std::vector<AbstractBaseLayer *> layers;
  std::function<double(Eigen::VectorXd, Eigen::VectorXd)> loss;
  std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd)> lossPrime;
};

#endif