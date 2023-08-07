#include <array>
#include <cmath>
#include <iostream>
#include <stdlib.h>

// be careful to set numeric values as floats here
#define PARTICLES 300              // number of particles
#define DIMENSION 1                // dimension
#define PARTICLE_MASS 10.0         // mass of a particle
#define LJ_CUTOFF_DISTANCE 0.00001 // LJ explodes for very close particles, stop earlier
#define RADIAL_HISTOGRAM_BINS 20   // into how many radius boxes we aggregate particles
#define VELOCITY_HISTOGRAM_BINS 12 // similarly, number of bins for the velocity histogram
#define HISTOGRAM_AVERAGE_N 20     // histogram averaging

struct Parameters {
  double alpha = 2.0;
  double beta = 1.5;
  double initWindowLength = 0.5;
  double tau = 12.0e-4;
};

#define square(x) ((x) * (x))

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
 protected:
  Parameters params;
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
    return sum;
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
  void initRandomly();
  void simulate(size_t timesteps, bool dot = false);
  void f(ParticleVectors &accelerations);
  void reflectParticles();
  double getKineticEnergy();
  double getLJPotential();
  double getTotalEnergy();
  void computeRadiusHistogram();
  void computeVelocityHistogram();
  void exportToCSV();
};
