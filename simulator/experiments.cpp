#include "Box.h"

void simpleExperiment() {
  auto box = ParticleBox();
  box.initRandomly();
  box.simulate(10000);

  double minimalPotential = box.getLJPotential();
  for (size_t i = 0; i < 8000; i++) {
    box.simulate(2);
    double potential = box.getLJPotential();
    if (potential < minimalPotential) {
      box.computeRadiusHistogram();
      minimalPotential = potential;
      box.exportToCSV();
      std::cout << "new min!" << std::endl;
    }
    std::cout << potential << std::endl;
  }

  // for (size_t i = 0; i < 10; i++) {
  //   box.simulate(1);
  //   box.computeRadiusHistogram();
  // }
  // box.exportToCSV();
}

int main() {
  srand(time(NULL));
  simpleExperiment();
  return 0;
}
