#include "Simulator.h"

int main(int argc, char **argv) {
  QApplication app(argc, argv);
  setlocale(LC_NUMERIC, "en_US.UTF-8");
  srand(time(NULL));

  BoxSimulator simulator;
  if (simulator.setupFromArgv(argc, argv) != 0)
    return 1;
  simulator.initRandomly();
  simulator.buildUI();
  simulator.resize(1380, 892);
  simulator.show();
  return app.exec();
}
