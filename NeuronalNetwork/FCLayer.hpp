#ifndef FCLAYER_HPP
#define FCLAYER_HPP

#include <fstream>
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
   * @param name_ Optional name for this layer
   */
  FCLayer(unsigned int input_size, unsigned int output_size,
          std::string name_ = "") {
    weights = Eigen::MatrixXd::Random(input_size, output_size);
    bias = Eigen::MatrixXd::Random(output_size, 1);
    name = name_;
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

  /**
   * @copydoc AbstractBaseLayer::saveConfiguration
   */
  void saveConfiguration(std::string path) const {
    saveWeights(path + name + "Weights.txt");
    saveBias(path + name + "Bias.txt");
  }

  /**
   * @copydoc AbstractBaseLayer::loadConfiguration
   */
  void loadConfiguration(std::string path) {
    loadWeights(path + name + "Weights.txt");
    loadBias(path + name + "Bias.txt");
  }

  /**
   * @brief Saves weights to disk
   * @param filename Where to write
   */
  void saveWeights(std::string filename) const {
    std::ofstream out(filename);
    for (unsigned int col = 0; col < weights.cols(); col++) {
      for (unsigned int row = 0; row < weights.rows(); row++) {
        out << weights(row, col) << " ";
      }
      out << std::endl;
    }
    out.close();
  }

  /**
   * @brief Saves bias to disk
   * @param filename Where to write
   */
  void saveBias(std::string filename) const {
    std::ofstream out(filename);
    for (unsigned int row = 0; row < bias.size(); row++) {
      out << bias(row) << " ";
    }
    out << std::endl;

    out.close();
  }

  /**
   * @brief Loads weights from disk
   * @param filename Where to write
   */
  void loadWeights(std::string filename) {
    std::ifstream in(filename);
    double value;
    for (unsigned int col = 0; col < weights.cols(); col++) {
      for (unsigned int row = 0; row < weights.rows(); row++) {
        in >> value;
        weights(row, col) = value;
      }
    }
    in.close();
  }

  /**
   * @brief Loads bias from disk
   * @param filename Where to write
   */
  void loadBias(std::string filename) {
    std::ifstream in(filename);
    double value;
    for (unsigned int row = 0; row < bias.size(); row++) {
      in >> value;
      bias(row) = value;
    }
    in.close();
  }

 private:
  Eigen::MatrixXd weights;  ///< internal weights
  Eigen::VectorXd bias;     ///< internal bias
  std::string name;         ///< internal name
};

#endif