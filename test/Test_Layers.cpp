#include <NeuronalNetwork/All>

int main(int argc, char const* argv[]) {
  FCLayer layer1(1, 1, "First Layer");
  ActivationLayer activ([](double x) { return x; }, [](double x) { return x; });
}