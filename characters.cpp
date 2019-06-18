#include <fstream>
#include "NeuronalNetwork/includes.hpp"

void readFileToMatrix(Eigen::MatrixXd& input, const char* filename, double divisor) {
  // lines: how many lines there should be after reading
  // cols: how many columns the matrix has (=input nodes)
  std::ifstream in(filename);
  int value;
  for (unsigned line = 0; line < input.rows(); line++) {
    for (unsigned int i = 0; i < input.cols(); i++) {
      in >> value;
      input(line, i) = ((double)value) / divisor;
    }
  }
  in.close();
}

int main() {
  std::cout << "This file must be run from the root directory for the files to "
               "load properly!"
            << std::endl;
  auto activation = [](double x) -> double { return tanh(x); };
  auto activationPrime = [](double x) -> double {
    return 1 - tanh(x) * tanh(x);
  };

  auto loss = [](Eigen::VectorXd y_true, Eigen::VectorXd y_pred) -> double {
    return (y_pred - y_true).squaredNorm() / ((double)y_true.size());
  };
  auto loss_prime = [](Eigen::VectorXd y_true,
                       Eigen::VectorXd y_pred) -> Eigen::VectorXd {
    return 2.0 / ((double)y_true.size()) * (y_pred - y_true);
  };

  unsigned int lines = 1000;
  unsigned int test_lines = 10;
  unsigned int epochs = 10;
  double learning_rate = 0.1;

  Eigen::MatrixXd x_data = Eigen::MatrixXd::Zero(lines, 28 * 28);
  Eigen::MatrixXd y_data = Eigen::MatrixXd::Zero(lines, 1);
  readFileToMatrix(x_data, "./test/images.txt",255.);
  readFileToMatrix(y_data, "./test/labels.txt",10.);

  Eigen::MatrixXd x_test = Eigen::MatrixXd::Zero(test_lines, 28 * 28);
  Eigen::MatrixXd y_test = Eigen::MatrixXd::Zero(test_lines, 1);
  readFileToMatrix(x_test, "./test/test_images.txt",255.);
  readFileToMatrix(y_test, "./test/test_labels.txt",10.);

  Network net = Network();
  FCLayer first(x_data.cols(), 300,"first");
  ActivationLayer a1(activation, activationPrime);
  FCLayer second(300, 1,"second");
  ActivationLayer a2(activation, activationPrime);

  net.add(&first);
  net.add(&a1);
  net.add(&second);
  net.add(&a2);

  net.use(loss, loss_prime);
  std::cout << "Starting training..." << std::endl;
  net.saveConfiguration("models/zero_");
  net.fit(x_data, y_data, epochs, learning_rate);
  net.saveConfiguration("models/first_");
  net.fit(x_data, y_data, 1, learning_rate);
  net.saveConfiguration("models/second_");
  std::cout << "Done. Predicting..." << std::endl;
  Eigen::MatrixXd out = net.predict(x_test);
  std::cout << "Predicted: " << out.transpose() << std::endl;
  std::cout << "True:      " << y_test.transpose() << std::endl;

  return 0;
}