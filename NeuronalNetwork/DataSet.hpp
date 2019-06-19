#ifndef DATASET_HPP
#define DATASET_HPP

#include <eigen3/Eigen/Dense>
#include <fstream>

/**
 * @brief Internal representation of data for the Network
 */
class DataSet {
 public:
  /**
   * @brief Default constructor
   */
  DataSet() = default;

  /**
   * @brief Defautl constructor
   * @param input_size Size of input vector
   * @param output_size Size of output vector
   */
  DataSet(const unsigned int input_size, const unsigned int output_size) {
    input_train_ = Eigen::MatrixXd::Zero(0, input_size);
    output_train_ = Eigen::MatrixXd::Zero(0, output_size);
    input_test_ = Eigen::MatrixXd::Zero(0, input_size);
  }

  /**
   * @brief Add a training set to the Network
   * @param input Input Vector
   * @param output Output Vector (=input passed through the network)
   */
  void addTrainingSet(const Eigen::VectorXd& input,
                      const Eigen::VectorXd& output) {
    Eigen::MatrixXd new_input =
        Eigen::MatrixXd(input_train_.rows() + 1, input_train_.cols());
    Eigen::MatrixXd new_output =
        Eigen::MatrixXd(output_train_.rows() + 1, output_train_.cols());
    new_input.topRows(input_train_.rows()) = input_train_;
    new_output.topRows(output_train_.rows()) = output_train_;
    new_input.bottomRows(1) = input.transpose();
    new_output.bottomRows(1) = output.transpose();
    input_train_ = new_input;
    output_train_ = new_output;
  }

  /**
   * @brief Add a test set to the Network
   * @param input Vector to pass through the network
   */
  void addTestSet(const Eigen::VectorXd& input) {
    Eigen::MatrixXd new_input =
        Eigen::MatrixXd(input_test_.rows() + 1, input_test_.cols());
    new_input.topRows(input_test_.rows()) = input_test_;
    new_input.bottomRows(1) = input.transpose();
    input_test_ = new_input;
  }

  /**
   * @brief Read the dataset from file
   * @param path Path to file
   * @note Format of file: Training Rows, Input Size, Output Size, Test
   * Rows, Input Training (rowwise), Output Training (rowwise), Input Test
   * (rowwise)
   */
  void readFromFile(const std::string& path) {
    std::ifstream in(path);
    unsigned int trainingRows, inputSize, outputSize, testRows;
    in >> trainingRows >> inputSize >> outputSize >> testRows;
    input_train_ = Eigen::MatrixXd(trainingRows, inputSize);
    output_train_ = Eigen::MatrixXd(trainingRows, outputSize);
    input_test_ = Eigen::MatrixXd(testRows, inputSize);

    // Read input Training
    double value;
    for (unsigned int i = 0; i < trainingRows; i++) {
      for (unsigned int j = 0; j < inputSize; j++) {
        in >> value;
        input_train_(i, j) = value;
      }
    }
    // Read output training
    for (unsigned int i = 0; i < trainingRows; i++) {
      for (unsigned int j = 0; j < outputSize; j++) {
        in >> value;
        output_train_(i, j) = value;
      }
    }
    // read input test
    for (unsigned int i = 0; i < testRows; i++) {
      for (unsigned int j = 0; j < inputSize; j++) {
        in >> value;
        input_test_(i, j) = value;
      }
    }
    in.close();
  }
  /**
   * @brief Write the dataset to file
   * @param path Path to file
   * @note Format of file: Training Rows, Input Size, Output Size, Test
   * Rows, Input Training (rowwise), Output Training (rowwise), Input Test
   * (rowwise)
   */
  void writeToFile(const std::string& path) const {
    std::ofstream out(path);
    out << input_train_.rows() << " " << input_train_.cols() << " "
        << output_train_.cols() << " " << input_test_.rows() << std::endl
        << std::endl;
    for (unsigned int i = 0; i < input_train_.rows(); i++) {
      for (unsigned int j = 0; j < input_train_.cols(); j++) {
        out << input_train_(i, j) << " ";
      }
      out << std::endl;
    }
    out << std::endl;
    // Read output training
    for (unsigned int i = 0; i < input_train_.rows(); i++) {
      for (unsigned int j = 0; j < output_train_.cols(); j++) {
        out << output_train_(i, j) << " ";
      }
      out << std::endl;
    }
    out << std::endl;
    // read input test
    for (unsigned int i = 0; i < input_test_.rows(); i++) {
      for (unsigned int j = 0; j < input_train_.cols(); j++) {
        out << input_test_(i, j) << " ";
      }
      out << std::endl;
    }
    out.close();
  }

  /**
   * @brief Getter for input training
   * @return Input training data as a matrix
   */
  Eigen::MatrixXd getInputTrainingData() const { return input_train_; }

  /**
   * @brief Getter for output training
   * @return Output training data as a matrix
   */
  Eigen::MatrixXd getOutputTrainingData() const { return output_train_; }

  /**
   * @brief Getter for input test
   * @return Input test data as a matrix
   */
  Eigen::MatrixXd getInputTestData() const { return input_test_; }

 private:
  Eigen::MatrixXd input_train_;   ///< input training data
  Eigen::MatrixXd output_train_;  ///< output training data
  Eigen::MatrixXd input_test_;    ///< input test data
};

#endif