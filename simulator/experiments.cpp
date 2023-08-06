#include "Box.h"

void simpleExperiment() {
  auto box = ParticleBox();
  box.initRandomly();
  box.simulate(1000);

  for (size_t i = 0; i < 10; i++) {
    box.simulate(5);
    box.computeRadiusHistogram();
  }
  box.exportToCSV();
}

int main() {
  simpleExperiment();
  return 0;
}
