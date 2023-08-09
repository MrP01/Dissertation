#include "Box.h"
#include <string.h>

int main(int argc, char *argv[]) {
  srand(time(NULL));
  auto box = ParticleBox();
  if (box.setupFromArgv(argc, argv) != 0)
    return 1;

  size_t iterations = (10000 * (PARTICLES / 200.0));
  if (argc >= 4)
    iterations = atol(argv[3]);

  box.initRandomly();
  box.simulate(iterations, true);

  double minimalPotential = box.getLJPotential();
  box.exportToCSV();
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
