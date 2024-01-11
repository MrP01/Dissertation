#include "Box.h"
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QLabel>
#include <QMainWindow>
#include <QPushButton>
#include <QShortcut>
#include <QSlider>
#include <QTimer>
#include <QtCharts/QBarCategoryAxis>
#include <QtCharts/QBarSeries>
#include <QtCharts/QBarSet>
#include <QtCharts/QChartView>
#include <QtCharts/QHorizontalStackedBarSeries>
#include <QtCharts/QLineSeries>
#include <QtCharts/QLogValueAxis>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QStackedBarSeries>
#include <QtCharts/QValueAxis>

#define STEPS_PER_FRAME 2        // number of timesteps per frame
#define FRAMES_PER_MEASUREMENT 2 // how often we measure
#define STEPS_PER_MEASUREMENT (STEPS_PER_FRAME * FRAMES_PER_MEASUREMENT)
#define MEASUREMENTS_IN_ENERGY_PLOT 800

class BoxSimulator : public ParticleBox, public QMainWindow {
 private:
  QChart *particleChart = new QChart();
  QScatterSeries *particleSeries = new QScatterSeries();

  QPixmap fish;
  QPixmap loonie;
  bool fishView = true;

  QChart *energyChart = new QChart();
  QLineSeries *kineticEnergySeries = new QLineSeries();
  QLineSeries *potentialEnergySeries = new QLineSeries();
  QLineSeries *LJpotentialEnergySeries = new QLineSeries();
  QLineSeries *totalEnergySeries = new QLineSeries();

  QChart *radiusHistChart = new QChart();
  QBarSet *radiusHistSet = new QBarSet("Radial Distance Histogram");
  QChart *velocityHistChart = new QChart();
  QBarSet *velocityHistSet = new QBarSet("Velocity Histogram");
  QChart *phaseSpaceChart = new QChart();
  QScatterSeries *phaseSpaceSeries = new QScatterSeries();
  QChart *phaseSpaceChart2 = new QChart();
  QScatterSeries *phaseSpaceSeries2 = new QScatterSeries();

  QPushButton *stepBtn = new QPushButton("Step");
  QPushButton *controlBtn = new QPushButton("Start");
  QPushButton *liftBtn = new QPushButton("Lift Up");
  QPushButton *slowDownBtn = new QPushButton("Slow Down");
  QPushButton *bringDownBtn = new QPushButton("Bring Down");
  QPushButton *reinitBtn = new QPushButton("Re-init");
  QPushButton *exportBtn = new QPushButton("Export");
  QLabel *statsLabel = new QLabel(QString("Particles in a box from [-1, 1]."));

  size_t _step = 0;
  size_t _start_step = 0;
  double _energyMax = 0;
  size_t _timerId;
  void renderParticles();
  void updateHistograms();
  void measure();
  void step();
  void timerEvent(QTimerEvent *event);

  void paintEvent(QPaintEvent *event);

  // very important:
  void setTheme(QChart::ChartTheme theme);

 public:
  BoxSimulator() = default;
  void buildUI();
};
