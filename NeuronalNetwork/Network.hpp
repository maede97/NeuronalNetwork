#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <cassert>
#include <functional>
#include <iostream>
#include <vector>

#include "AbstractBaseLayer.hpp"
#include "DataSet.hpp"

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
  void saveConfiguration(const std::string& path) const {
    for (auto layer = layers_.begin(); layer != layers_.end(); layer++) {
      (*layer)->saveConfiguration(path);
    }
  }
  /**
   * @brief Loads the whole Network configuration to disk
   * @param path Folder to be loaded from
   */
  void loadConfiguration(const std::string& path) {
    for (auto layer = layers_.begin(); layer != layers_.end(); layer++) {
      (*layer)->loadConfiguration(path);
    }
  }

  /**
   * @brief Add a new layer
   * @param layer Pointer to the layer
   */
  void add(AbstractBaseLayer *layer) { layers_.push_back(layer); }
  /**
   * @brief Sets loss function and derivative
   * @param loss Loss function
   * @param lossPrime The derivative of the loss functoin
   */
  void use(std::function<double(Eigen::VectorXd, Eigen::VectorXd)> loss,
           std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd)>
               lossPrime) {
    loss_ = loss;
    lossPrime_ = lossPrime;
  }

  /**
   * @brief Set DataSet
   * @param set The DataSet to use
   */
  void setData(const DataSet& set) { set_ = set; }
  
  /**
   * @brief Returns the last error.
   * @return The last error.
   */
  double lastError() const { return last_err_; }

  /**
   * @brief Predict output of data
   * @return A Matrix containing all outputs as rows
   */
  Eigen::MatrixXd predict() const {
    const unsigned int samples = set_.getInputTestData().rows();
    std::vector<Eigen::VectorXd> result;
    Eigen::VectorXd output;
    for (unsigned int i = 0; i < samples; i++) {
      output = set_.getInputTestData().row(i).transpose();
      for (auto layer = layers_.begin(); layer != layers_.end(); layer++) {
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
   * @param epochs How long to train
   * @param learning_rate With which rate to train
   */
  void fit(const unsigned int epochs, const double learning_rate) {
    assert(loss_ && lossPrime_ && "Loss function not initialised.");
    const unsigned int samples = set_.getInputTrainingData().rows();
    Eigen::VectorXd output;
    for (unsigned int i = 0; i < epochs; i++) {
      double err = 0;
      for (unsigned int j = 0; j < samples; j++) {
        output = set_.getInputTrainingData().row(j).transpose();
        for (auto layer = layers_.begin(); layer != layers_.end(); layer++) {
          output = (*layer)->forwardPropagation(output);
        }
        // compute for display purpose
        err += loss_(set_.getOutputTrainingData().row(j).transpose(), output);

        // backwards propagation
        Eigen::VectorXd error =
            lossPrime_(set_.getOutputTrainingData().row(j).transpose(), output);
        for (auto layer = layers_.end() - 1; layer + 1 != layers_.begin();
             layer--) {
          error = (*layer)->backwardPropagation(error, learning_rate);
        }
      }
      err /= (double)samples;
      last_err_ = err;
    }
    std::cout << std::endl;
  }

 private:
  std::vector<AbstractBaseLayer *> layers_;  ///< internal storage for layers
  std::function<double(Eigen::VectorXd, Eigen::VectorXd)>
      loss_;  ///< loss function to use
  std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd)>
      lossPrime_;  ///< loss derivative to use
  DataSet set_;
  double last_err_; ///< last error
};

#endif
