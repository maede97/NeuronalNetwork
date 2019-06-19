#include <NeuronalNetwork/All>
#include <cassert>
void initializer() {
  DataSet data(3, 2);
  DataSet d;
  DataSet data2(6, 1);
}

void trainingData() {
  DataSet data(3, 2);
  data.addTrainingSet(Eigen::Vector3d(1, 2, 3), Eigen::Vector2d(4, 2));
  data.addTrainingSet(Eigen::Vector3d(3, 2, 1), Eigen::Vector2d(2, 3));
}

void testData() {
  DataSet data(3, 2);
  data.addTestSet(Eigen::Vector3d(1, 2, 3));
  data.addTestSet(Eigen::Vector3d(3, 2, 1));
}
void allData() {
  DataSet data(3, 2);
  data.addTestSet(Eigen::Vector3d(1, 2, 3));
  data.addTestSet(Eigen::Vector3d(3, 2, 1));
  data.addTrainingSet(Eigen::Vector3d(1, 2, 3), Eigen::Vector2d(4, 2));
  data.addTrainingSet(Eigen::Vector3d(3, 2, 1), Eigen::Vector2d(2, 3));
}
void checkMatrices() {
  DataSet data(3, 2);
  data.addTestSet(Eigen::Vector3d(1, 2, 3));
  data.addTestSet(Eigen::Vector3d(3, 2, 1));
  data.addTrainingSet(Eigen::Vector3d(1, 2, 3), Eigen::Vector2d(4, 2));
  data.addTrainingSet(Eigen::Vector3d(3, 2, 1), Eigen::Vector2d(2, 3));
  assert(data.getInputTestData().rows() == 2 &&
         data.getInputTestData().cols() == 3);
  assert(data.getInputTrainingData().rows() == 2 &&
         data.getInputTrainingData().cols() == 3);
  assert(data.getOutputTrainingData().rows() == 2 &&
         data.getOutputTrainingData().cols() == 2);
  Eigen::MatrixXd inputTraining(2,3), outputTraining(2,2), inputTest(2,3);
  inputTraining << 1,2,3,3,2,1;
  outputTraining << 4,2,2,3;
  inputTest << 1, 2, 3, 3, 2, 1;
  assert(inputTraining == data.getInputTrainingData());
  assert(outputTraining == data.getOutputTrainingData());
  assert(inputTest == data.getInputTestData());
}

int main() {
  initializer();
  trainingData();
  testData();
  allData();
  checkMatrices();

  return 0;
}