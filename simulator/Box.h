#include <array>
#include <cmath>
#include <iostream>
#include <stdlib.h>

// be careful to set numeric values as floats here
#define PARTICLES 120              // number of particles
#define DIMENSION 1                // dimension
#define PARTICLE_MASS 1.0          // mass of a particle
#define LJ_CUTOFF_DISTANCE 0.00001 // LJ explodes for very close particles, stop earlier
#define RADIAL_HISTOGRAM_BINS 20   // into how many radius boxes we aggregate particles
#define VELOCITY_HISTOGRAM_BINS 12 // similarly, number of bins for the velocity histogram
#define HISTOGRAM_AVERAGE_N 20     // histogram averaging

class InteractionPotential {
 public:
  virtual double potential(double r) { return 0.0; };
  virtual double force(double r) { return 0.0; };
};

class AttractiveRepulsive : public InteractionPotential {
 public:
  double alpha = 2.0;
  double beta = 1.5;
  double potential(double r) { return pow(r, alpha) / alpha - pow(r, beta) / beta; }
  double force(double r) { return pow(r, alpha - 1.0) - pow(r, beta - 1.0); }
};

class MorsePotential : public InteractionPotential {
 public:
  double C_att = 1.5;
  double l_att = 2.0;
  double C_rep = 1.0;
  double l_rep = 0.5;
  double potential(double r) { return C_rep * exp(-r / l_rep) - C_att * exp(-r / l_att); };
  double force(double r) { return C_att / l_att * exp(-r / l_att) - C_rep / l_rep * exp(-r / l_rep); };
};

struct Parameters {
  double tau = 20.0e-4;          // time step
  double boxScaling = 4.0;       // size of the box: [-1, 1] * boxScaling
  double initWindowLength = 1.0; // 0.0 < window length <= 2.0
  double selfPropulsion = 1.6;   // "alpha" parameter in 2006-self-propelled
  double friction = 0.5;         // "beta" parameter in 2006-self-propelled
};

#define square(x) ((x) * (x))
#define sign(x) (((x) >= 0) ? 1.0 : -1.0)

using ParticleVectors = double (&)[PARTICLES][DIMENSION];

struct RadiusHistogram {
  double min, max;
  float heights[RADIAL_HISTOGRAM_BINS] = {0};
  float maxHeight = 1;
};

struct VelocityHistogram {
  double min, max;
  size_t heights[RADIAL_HISTOGRAM_BINS] = {0};
  size_t maxHeight = 1;
};

class ParticleBox {
 public:
  Parameters p;
  InteractionPotential *interaction = new MorsePotential();
  double positions[PARTICLES][DIMENSION];
  double velocities[PARTICLES][DIMENSION];
  struct RadiusHistogram radiusHist;
  struct RadiusHistogram pastHistograms[HISTOGRAM_AVERAGE_N];
  struct RadiusHistogram averagedRadiusHistogram;
  struct VelocityHistogram velocityHist;
  double totalMeanVelocity = 0;

  double squaredDistanceBetween(size_t i, size_t j) {
    double sum = 0.0;
    for (size_t d = 0; d < DIMENSION; d++)
      sum += square(positions[i][d] - positions[j][d]);
    return square(p.boxScaling) * sum;
  }
  double distanceBetween(size_t i, size_t j) { return std::sqrt(squaredDistanceBetween(i, j)); }
  double totalVelocity(size_t i) {
    double sum = 0.0;
    for (size_t d = 0; d < DIMENSION; d++)
      sum += square(velocities[i][d]);
    return sqrt(sum);
  };

 public:
  ParticleBox() = default;
  int setupFromArgv(int argc, char **argv);
  void initRandomly();
  void simulate(size_t timesteps, bool dot = false);
  void f(ParticleVectors &accelerations);
  void reflectParticles();
  double friction(double v);
  double getKineticEnergy();
  double getLJPotential();
  double getTotalEnergy();
  void computeRadiusHistogram();
  void computeVelocityHistogram();
  void exportToCSV();
};
