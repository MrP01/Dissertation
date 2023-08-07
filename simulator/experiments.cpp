#include "Box.h"

int main() {
  srand(time(NULL));
  auto box = ParticleBox();
  box.initRandomly();
  box.simulate((size_t)(10000 * (PARTICLES / 200.0)), true);

  double minimalPotential = box.getLJPotential();
  size_t n = 200;
  while ((n--) > 0) {
    box.simulate(2);
    double potential = box.getLJPotential();
    if (potential < minimalPotential) {
      box.computeRadiusHistogram();
      minimalPotential = potential;
      box.exportToCSV();
      std::cout << "new min!" << std::endl;
      n = 200;
    }
    std::cout << n << ": " << potential << std::endl;
  }

  // for (size_t i = 0; i < 10; i++) {
  //   box.simulate(1);
  //   box.computeRadiusHistogram();
  // }
  // box.exportToCSV();
  return 0;
}
