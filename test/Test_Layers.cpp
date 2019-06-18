#include <NeuronalNetwork/All>

// used for activation layers
auto f = [](double x) { return x; };

void createLayers() {
  FCLayer layer1(1, 1, "A Layer");
  ActivationLayer activ(2, f, f);
  FCLayer layer2(4, 2);
}

void forwardPropagation() {
  FCLayer layer(2, 2, "Another Layer");
  ActivationLayer activ(2, f, f);
  layer.forwardPropagation(Eigen::Vector2d(3, 3));
  activ.forwardPropagation(Eigen::Vector2d(3, 3));
}

void backwardPropagation() {
  FCLayer layer(2, 5, "A third Layer");
  ActivationLayer activ(3, f, f);
  layer.backwardPropagation(Eigen::VectorXd::Random(5), 0.05);
  activ.backwardPropagation(Eigen::Vector3d::Random(), 1.4);
}

int main(int argc, char const* argv[]) {
  createLayers();
  forwardPropagation();
  backwardPropagation();
  return 0;
}