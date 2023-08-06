#include <array>
#include <cmath>
#include <iostream>
#include <stdlib.h>

// be careful to set numeric values as floats here
#define PARTICLES 100              // number of particles
#define DIMENSION 1                // dimension
#define INIT_WINDOW_LENGTH 1.0     // width of the initialisation interval for particles (default [-1, 1], so width 2)
#define PARTICLE_MASS 10.0         // mass of a particle
#define LJ_CUTOFF_DISTANCE 0.00001 // LJ explodes for very close particles, stop earlier
#define TAU 3.0e-3                 // time step
#define RADIAL_HISTOGRAM_BINS 20   // into how many radius boxes we aggregate particles
#define VELOCITY_HISTOGRAM_BINS 12 // similarly, number of bins for the velocity histogram
#define HISTOGRAM_AVERAGE_N 200    // histogram averaging
#define ALPHA 2.5                  // (attractive) parameter alpha for the kernel K(r)
#define BETA 1.6                   // (repulsive) parameter beta for the kernel K(r)

#define square(x) ((x) * (x))

#define LJ_CUT_DIST_SQ (LJ_CUTOFF_DISTANCE * LJ_CUTOFF_DISTANCE)

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

 public:
  ParticleBox() = default;
  void initRandomly();
  void simulate(size_t timesteps);
  void f(ParticleVectors &accelerations);
  void reflectParticles();
  double getKineticEnergy();
  double getLJPotential();
  double getTotalEnergy();
  void computeRadiusHistogram();
  void computeVelocityHistogram();
  void exportToCSV();
};
