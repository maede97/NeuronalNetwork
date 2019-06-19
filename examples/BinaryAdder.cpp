#include <NeuronalNetwork/All>

int main(int argc, char const* argv[]) {
  // This demonstrates a simple binary adder

  // create activation and derivative
  auto activation = [](double x) -> double { return tanh(x); };
  auto activationPrime = [](double x) -> double {
    return 1 - tanh(x) * tanh(x);
  };

  // create loss and derivative
  auto loss = [](Eigen::VectorXd y_true, Eigen::VectorXd y_pred) -> double {
    return (y_pred - y_true).squaredNorm() / ((double)y_true.size());
  };
  auto loss_prime = [](Eigen::VectorXd y_true,
                       Eigen::VectorXd y_pred) -> Eigen::VectorXd {
    return 2.0 / ((double)y_true.size()) * (y_pred - y_true);
  };

  // Create training data
  DataSet data(4,3);
  data.addTrainingSet(Eigen::Vector4d(0, 0, 0, 0), Eigen::Vector3d(0, 0, 0));
  data.addTrainingSet(Eigen::Vector4d(0, 0, 0, 1), Eigen::Vector3d(0, 0, 1));
  data.addTrainingSet(Eigen::Vector4d(0, 0, 1, 0), Eigen::Vector3d(0, 1, 0));
  data.addTrainingSet(Eigen::Vector4d(0, 0, 1, 1), Eigen::Vector3d(0, 1, 1));
  data.addTrainingSet(Eigen::Vector4d(0, 1, 0, 0), Eigen::Vector3d(0, 0, 1));
  data.addTrainingSet(Eigen::Vector4d(0, 1, 0, 1), Eigen::Vector3d(0, 1, 0));
  data.addTrainingSet(Eigen::Vector4d(0, 1, 1, 0), Eigen::Vector3d(0, 1, 1));
  data.addTrainingSet(Eigen::Vector4d(0, 1, 1, 1), Eigen::Vector3d(1, 0, 0));
  data.addTrainingSet(Eigen::Vector4d(1, 0, 0, 0), Eigen::Vector3d(0, 1, 0));
  data.addTrainingSet(Eigen::Vector4d(1, 0, 0, 1), Eigen::Vector3d(0, 1, 1));
  data.addTrainingSet(Eigen::Vector4d(1, 0, 1, 0), Eigen::Vector3d(1, 0, 0));
  data.addTrainingSet(Eigen::Vector4d(1, 0, 1, 1), Eigen::Vector3d(1, 0, 1));
  data.addTrainingSet(Eigen::Vector4d(1, 1, 0, 0), Eigen::Vector3d(0, 1, 1));
  data.addTrainingSet(Eigen::Vector4d(1, 1, 0, 1), Eigen::Vector3d(1, 0, 0));
  data.addTrainingSet(Eigen::Vector4d(1, 1, 1, 0), Eigen::Vector3d(1, 0, 1));
  data.addTrainingSet(Eigen::Vector4d(1, 1, 1, 1), Eigen::Vector3d(1, 1, 0));

  data.addTestSet(Eigen::Vector4d(0, 1, 1, 0));

  //data.writeToFile("binaryAdder.txt");

  // Create Layers
  FCLayer input_layer(4, 6, "Input");
  ActivationLayer activation1(6, activation, activationPrime);
  FCLayer output_layer(6, 3, "Output");
  ActivationLayer activation2(3, activation, activationPrime);

  // Create Netwrok
  Network net = Network();
  net.use(loss, loss_prime);
  net.setData(data);

  // Add layers
  net.add(&input_layer);
  net.add(&activation1);
  net.add(&output_layer);
  net.add(&activation2);

  net.fit(1000, 0.1);

  Eigen::MatrixXd net_predict = net.predict();
  std::cout << "Network Output:" << std::endl << net_predict << std::endl;
  std::cout << "Original Output:" << std::endl
            << Eigen::Vector3d(0, 1, 1).transpose() << std::endl;

  return 0;
}