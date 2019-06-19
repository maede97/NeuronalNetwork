#ifndef FCLAYER_HPP
#define FCLAYER_HPP

#include <fstream>
#include "AbstractBaseLayer.hpp"

/**
 * @brief Fully Connected Layer
 */
class FCLayer : public AbstractBaseLayer {
 public:
  /**
   * @brief Create FCLayer
   * @param input_size How many inputs there are
   * @param output_size How many outputs there are
   * @param name Optional name for this layer
   */
  FCLayer(const unsigned int input_size, const unsigned int output_size,
          std::string name = "") {
    // set random and default vectors
    weights_ = Eigen::MatrixXd::Random(input_size, output_size);
    bias_ = Eigen::MatrixXd::Random(output_size, 1);
    input_ = Eigen::VectorXd::Zero(input_size);
    name_ = name;
  }

  /**
   * @copydoc AbstractBaseLayer::forwardPropagation
   */
  Eigen::VectorXd forwardPropagation(const Eigen::VectorXd& input_data) {
    input_ = input_data;
    return weights_.transpose() * input_ + bias_;
  }

  /**
   * @copydoc AbstractBaseLayer::backwardPropagation
   */
  Eigen::VectorXd backwardPropagation(const Eigen::VectorXd& output_error,
                                      const double learningRate) {
    Eigen::VectorXd input_error =
        output_error.transpose() * weights_.transpose();
    Eigen::MatrixXd weights_error = input_ * output_error.transpose();
    weights_ -= learningRate * weights_error;
    bias_ -= learningRate * output_error;
    return input_error;
  }

  /**
   * @copydoc AbstractBaseLayer::saveConfiguration
   */
  void saveConfiguration(const std::string& path) const {
    saveWeights(path + name_ + "Weights.txt");
    saveBias(path + name_ + "Bias.txt");
  }

  /**
   * @copydoc AbstractBaseLayer::loadConfiguration
   */
  void loadConfiguration(const std::string& path) {
    loadWeights(path + name_ + "Weights.txt");
    loadBias(path + name_ + "Bias.txt");
  }

  /**
   * @brief Saves weights to disk
   * @param filename Where to write
   */
  void saveWeights(const std::string& filename) const {
    std::ofstream out(filename);
    for (unsigned int col = 0; col < weights_.cols(); col++) {
      for (unsigned int row = 0; row < weights_.rows(); row++) {
        out << weights_(row, col) << " ";
      }
      out << std::endl;
    }
    out.close();
  }

  /**
   * @brief Saves bias to disk
   * @param filename Where to write
   */
  void saveBias(const std::string& filename) const {
    std::ofstream out(filename);
    for (unsigned int row = 0; row < bias_.size(); row++) {
      out << bias_(row) << " ";
    }
    out << std::endl;

    out.close();
  }

  /**
   * @brief Loads weights from disk
   * @param filename Where to write
   */
  void loadWeights(const std::string& filename) {
    std::ifstream in(filename);
    double value;
    for (unsigned int col = 0; col < weights_.cols(); col++) {
      for (unsigned int row = 0; row < weights_.rows(); row++) {
        in >> value;
        weights_(row, col) = value;
      }
    }
    in.close();
  }

  /**
   * @brief Loads bias from disk
   * @param filename Where to write
   */
  void loadBias(const std::string& filename) {
    std::ifstream in(filename);
    double value;
    for (unsigned int row = 0; row < bias_.size(); row++) {
      in >> value;
      bias_(row) = value;
    }
    in.close();
  }

 private:
  Eigen::MatrixXd weights_;  ///< internal weights
  Eigen::VectorXd bias_;     ///< internal bias
  std::string name_;         ///< internal name
};

#endif