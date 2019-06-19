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
  FCLayer layer1(2, 4, "Another Layer");
  ActivationLayer activ(4, f, f);
  FCLayer layer2(4, 2);
  DataSet data(2,2);
  data.addTestSet(Eigen::Vector2d(1, 1));
  data.addTrainingSet(Eigen::Vector2d(1, 1), Eigen::Vector2d(0, 1));
  Network net = Network();
  net.setData(data);
  net.add(&layer1);
  net.add(&activ);
  net.add(&layer2);
  net.predict();
}

void fit() {
  FCLayer layer1(2, 4, "A third Layer");
  ActivationLayer activ(4, f, f);
  FCLayer layer2(4, 2);

  DataSet data(2,2);
  data.addTestSet(Eigen::Vector2d(1, 1));
  data.addTrainingSet(Eigen::Vector2d(1, 1), Eigen::Vector2d(0, 1));

  Network net = Network();
  net.setData(data);
  net.use([](Eigen::VectorXd, Eigen::VectorXd) -> double { return 1.0; },
          [](Eigen::VectorXd x, Eigen::VectorXd y) -> Eigen::VectorXd {
            return x;
          });
  net.add(&layer1);
  net.add(&activ);
  net.add(&layer2);
  net.fit(1, 0.2);
}

int main(int argc, char const* argv[]) {
  createNetwork();
  predict();
  fit();
  return 0;
}