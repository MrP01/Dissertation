#include "Box.h"
#include <algorithm>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <string.h>

void ParticleBox::initRandomly(double initialKineticEnergy, double initialGravitationalPotential) {
  double approxRadius = initialGravitationalPotential / (PARTICLE_MASS * GRAVITY);
  for (size_t i = 0; i < PARTICLES; i++) {
    double closestNeighbourDist = 0;
    while (closestNeighbourDist < 0.8 * LJ_SIGMA) {
      for (size_t d = 0; d < DIMENSION; d++)
        positions[i][d] = ((double)rand() / RAND_MAX - 0.5) * INIT_WINDOW_LENGTH;
      closestNeighbourDist = LJ_SIGMA;
      for (size_t j = 0; j < i; j++)
        closestNeighbourDist = std::min(closestNeighbourDist, distanceBetween(i, j));
      std::cout << "." << std::flush;
    }

    std::cout << "Init particle at " << positions[i][0] << std::endl;
    for (size_t d = 0; d < DIMENSION; d++)
      velocities[i][d] = (((double)rand() / RAND_MAX) - 0.5) / 2;
  }
}

void ParticleBox::f(ParticleVectors &accelerations) {
  for (size_t i = 0; i < PARTICLES; i++) {
    double forces[DIMENSION] = {0.0};
    for (size_t j = 0; j < PARTICLES; j++) {
      if (i == j)
        continue;
      double r_squared = squaredDistanceBetween(i, j);
      // std::cout << "r sq" << r_squared << std::endl;
      if (r_squared < LJ_CUT_DIST_SQ)
        r_squared = LJ_CUT_DIST_SQ;
      double r = sqrt(r_squared);
      for (size_t d = 0; d < DIMENSION; d++)
        forces[d] += (positions[i][d] - positions[j][d]) * (pow(r, ALPHA - 1) - pow(r, BETA - 1));
      // std::cout << (positions[i][0] - positions[j][0]) << std::endl;
    }
    // std::cout << "force " << forces[0] << std::endl;
    for (size_t d = 0; d < DIMENSION; d++)
      accelerations[i][d] = forces[d] / PARTICLE_MASS;
  }
}

void ParticleBox::simulate(size_t timesteps) {
  double after_accelerations[PARTICLES][DIMENSION];
  f(after_accelerations);
  for (size_t t = 0; t < timesteps; t++) {
    ParticleVectors before_accelerations = after_accelerations;
    memcpy(before_accelerations, after_accelerations, PARTICLES * DIMENSION * sizeof(double));
    for (size_t i = 0; i < PARTICLES; i++) {
      for (size_t d = 0; d < DIMENSION; d++)
        positions[i][d] += TAU * velocities[i][d] + square(TAU) / 2 * before_accelerations[i][d];
    }

    f(after_accelerations);
    // double totalVelocity = 0;
    for (size_t i = 0; i < PARTICLES; i++) {
      for (size_t d = 0; d < DIMENSION; d++) {
        velocities[i][d] += TAU / 2 * (before_accelerations[i][d] + after_accelerations[i][d]);
        // totalVelocity += square(velocities[i][d]);
      }
    }
    // totalMeanVelocity += totalVelocity / PARTICLES;
    // std::cout << "Position:" << positions[0][0] << ", " << positions[1][0] << ", " << positions[2][0] << std::endl;
    reflectParticles();
    // time += TIME_STEP
  }
}

void ParticleBox::reflectParticles() {
  for (size_t i = 0; i < PARTICLES; i++) {
    for (size_t d = 0; d < DIMENSION; d++) {
      if (positions[i][d] < -1.0) {
        positions[i][d] = -1.0 - (1.0 + positions[i][d]); // assumes linear movement in this timestep
        velocities[i][d] = -velocities[i][d];
      } else if (positions[i][d] > 1.0) {
        positions[i][d] = 1.0 - (positions[i][d] - 1.0); // assumes linear movement
        velocities[i][d] = -velocities[i][d];
      }
    }
  }
}

double ParticleBox::getKineticEnergy() {
  double energy = 0;
  for (size_t i = 0; i < PARTICLES; i++)
    for (size_t d = 0; d < DIMENSION; d++)
      energy += square(velocities[i][d]);
  std::cout << "Energy: " << energy << std::endl;
  return PARTICLE_MASS / 2 * energy;
}
double ParticleBox::getGravitationalPotential() {
  double totalRadius = 0;
  for (size_t i = 0; i < PARTICLES; i++)
    totalRadius += positions[i][1];
  // return PARTICLE_MASS * GRAVITY * totalRadius;
  return 0.0;
}
double ParticleBox::getLJPotential() {
  double energy = 0;
  for (size_t i = 0; i < PARTICLES; i++) {
    for (size_t j = 0; j < PARTICLES; j++) {
      if (i == j)
        continue;
      double r_squared = squaredDistanceBetween(i, j);
      if (r_squared < LJ_CUT_DIST_SQ)
        r_squared = LJ_CUT_DIST_SQ;
      double r = sqrt(r_squared);
      energy += std::pow(r, ALPHA) / ALPHA - std::pow(r, BETA) / BETA;
    }
  }
  return abs(energy);
}
double ParticleBox::getTotalEnergy() { return getKineticEnergy() + getGravitationalPotential() + getLJPotential(); }

void ParticleBox::computeRadiusHistogram() {
  radiusHist.min = 0;
  radiusHist.max = 1;
  const double delta = radiusHist.max - radiusHist.min;
  std::fill(std::begin(radiusHist.heights), std::end(radiusHist.heights), 0);
  for (size_t i = 0; i < PARTICLES; i++) {
    double r_squared = 0;
    for (size_t d = 0; d < DIMENSION; d++)
      r_squared += square(positions[i][d]);
    // TODO: the max radius is not 1 in a square domain
    size_t bin = floor((std::sqrt(r_squared) - radiusHist.min) / delta * RADIAL_HISTOGRAM_BINS);
    if (bin >= RADIAL_HISTOGRAM_BINS) // true for the last value
      bin = RADIAL_HISTOGRAM_BINS - 1;
    radiusHist.heights[bin]++;
  }
  radiusHist.maxHeight = 0;
  for (size_t bin = 0; bin < RADIAL_HISTOGRAM_BINS; bin++)
    radiusHist.maxHeight = std::max(radiusHist.maxHeight, radiusHist.heights[bin]);

  for (size_t i = HISTOGRAM_AVERAGE_N - 1; i > 0; i--)
    pastHistograms[i] = pastHistograms[i - 1];
  pastHistograms[0] = radiusHist;

  averagedRadiusHistogram.min = -1;
  averagedRadiusHistogram.max = 1;
  for (size_t j = 0; j < RADIAL_HISTOGRAM_BINS; j++)
    averagedRadiusHistogram.heights[j] = 0;
  for (size_t i = 0; i < HISTOGRAM_AVERAGE_N; i++) {
    for (size_t j = 0; j < RADIAL_HISTOGRAM_BINS; j++)
      averagedRadiusHistogram.heights[j] += pastHistograms[i].heights[j] / HISTOGRAM_AVERAGE_N;
  }
  averagedRadiusHistogram.maxHeight = 0;
  for (size_t bin = 0; bin < RADIAL_HISTOGRAM_BINS; bin++)
    averagedRadiusHistogram.maxHeight =
        std::max(averagedRadiusHistogram.maxHeight, averagedRadiusHistogram.heights[bin]);
}

void ParticleBox::computeVelocityHistogram() {
  std::array<double, PARTICLES> values;
  for (size_t i = 0; i < PARTICLES; i++) {
    for (size_t d = 0; d < DIMENSION; d++)
      values[i] += square(velocities[i][d]);
    values[i] = sqrt(values[i]);
  }
  const auto [_min, _max] = std::minmax_element(values.begin(), values.end());
  velocityHist.min = *_min;
  velocityHist.max = *_max;
  const double delta = velocityHist.max - velocityHist.min;
  std::fill(std::begin(velocityHist.heights), std::end(velocityHist.heights), 0);
  for (size_t i = 0; i < PARTICLES; i++) {
    size_t bin = floor((values[i] - velocityHist.min) / delta * RADIAL_HISTOGRAM_BINS);
    if (bin >= RADIAL_HISTOGRAM_BINS)
      bin = RADIAL_HISTOGRAM_BINS - 1;
    velocityHist.heights[bin]++;
  }
  velocityHist.maxHeight = 0;
  for (size_t bin = 0; bin < RADIAL_HISTOGRAM_BINS; bin++)
    velocityHist.maxHeight = std::max(velocityHist.maxHeight, velocityHist.heights[bin]);
}

void ParticleBox::exportToCSV() {
  std::ofstream positionsCsv("/tmp/positions.csv");
  for (size_t i = 0; i < PARTICLES; i++)
    positionsCsv << positions[i][0] << ", " << positions[i][1] << "\n";
  positionsCsv.close();
  std::ofstream velocitiesCsv("/tmp/velocities.csv");
  for (size_t i = 0; i < PARTICLES; i++)
    velocitiesCsv << velocities[i][0] << ", " << velocities[i][1] << "\n";
  velocitiesCsv.close();
  std::ofstream positionHistCsv("/tmp/position-histogram.csv");
  for (size_t i = 0; i < RADIAL_HISTOGRAM_BINS; i++)
    positionHistCsv << averagedRadiusHistogram.heights[i] << "\n";
  positionHistCsv.close();
}
