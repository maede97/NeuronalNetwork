#include <NeuronalNetwork/All>

// used for activation layers
auto f = [](double x) { return x; };
void createNetwork() {
  FCLayer layer1(1, 4, "A Layer");
  ActivationLayer activ(4, f, f);
  FCLayer layer2(4, 2);
  Network net = Network();
  net.add(&layer1);
  net.add(&activ);
  net.add(&layer2);
}

void predict() {
  FCLayer layer1(1, 4, "Another Layer");
  ActivationLayer activ(4, f, f);
  FCLayer layer2(4, 2);
  Network net = Network();
  net.add(&layer1);
  net.add(&activ);
  net.add(&layer2);
  Eigen::MatrixXd data(1, 1);
  data << 0.5;
  net.predict(data);
}

void fit() {
  FCLayer layer1(1, 4, "A third Layer");
  ActivationLayer activ(4, f, f);
  FCLayer layer2(4, 2);
  Network net = Network();

  net.use([](Eigen::VectorXd, Eigen::VectorXd) -> double { return 1.0; },
          [](Eigen::VectorXd x, Eigen::VectorXd y) -> Eigen::VectorXd {
            return x;
          });
  net.add(&layer1);
  net.add(&activ);
  net.add(&layer2);
  // fit 3 examples throught the network
  Eigen::MatrixXd dataX = Eigen::MatrixXd::Random(3, 1);
  Eigen::MatrixXd dataY = Eigen::MatrixXd::Random(3, 2);
  net.fit(dataX, dataY, 1, 0.2);
}

int main(int argc, char const* argv[]) {
  createNetwork();
  predict();
  fit();
  return 0;
}