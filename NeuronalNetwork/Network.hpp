#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <functional>
#include <iostream>
#include <vector>

#include "Layer.hpp"

/**
 * @brief Network class
 */
class Network {
 public:
  /**
   * @brief Default constructor
   */
  Network() {}

  /**
   * @brief Saves the whole Network configuration to disk
   * @param path Folder to be saved to
   */
  void saveConfiguration(std::string path) const {
    for (auto layer = layers.begin(); layer != layers.end(); layer++) {
      (*layer)->saveConfiguration(path);
    }
  }
  /**
   * @brief Loads the whole Network configuration to disk
   * @param path Folder to be loaded from
   */
  void loadConfiguration(std::string path) {
    for (auto layer = layers.begin(); layer != layers.end(); layer++) {
      (*layer)->loadConfiguration(path);
    }
  }

  /**
   * @brief Add a new layer
   * @param layer Pointer to the layer
   */
  void add(AbstractBaseLayer *layer) { layers.push_back(layer); }
  /**
   * @brief Sets loss function and derivative
   * @param loss_ Loss function
   * @param lossPrime_ The derivative of the loss functoin
   */
  void use(std::function<double(Eigen::VectorXd, Eigen::VectorXd)> loss_,
           std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd)>
               lossPrime_) {
    loss = loss_;
    lossPrime = lossPrime_;
  }

  /**
   * @brief Predict output of data
   * @param input_data Matrix containing each input data-set as a row
   * @return A Matrix containing all outputs as rows
   */
  Eigen::MatrixXd predict(Eigen::MatrixXd input_data) const {
    const unsigned int samples = input_data.rows();
    std::vector<Eigen::VectorXd> result;
    Eigen::VectorXd output;
    for (unsigned int i = 0; i < samples; i++) {
      output = input_data.row(i).transpose();
      for (auto layer = layers.begin(); layer != layers.end(); layer++) {
        output = (*layer)->forwardPropagation(output);
      }
      result.push_back(output);
    }
    Eigen::MatrixXd resultMat(samples, result[0].size());
    for (int i = 0; i < result.size(); i++) {
      resultMat.row(i) = result[i];
    }
    return resultMat;
  }

  /**
   * @brief Train the network
   * @param x_train Matrix containing all training input data as rows
   * @param y_train Matrix containing all training output data as rows
   * @param epochs How long to train
   * @param learning_rate With which rate to train
   */
  void fit(Eigen::MatrixXd x_train, Eigen::MatrixXd y_train,
           unsigned int epochs, double learning_rate) {
    const unsigned int samples = x_train.rows();
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
      std::cout << "Epoch: " << i + 1 << "\terror: " << err << "\r" << std::flush;
    }
    std::cout << std::endl;
  }

 private:
  std::vector<AbstractBaseLayer *> layers;  ///< internal storage for layers
  std::function<double(Eigen::VectorXd, Eigen::VectorXd)>
      loss;  ///< loss function to use
  std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd)>
      lossPrime;  ///< loss derivative to use
};

#endif