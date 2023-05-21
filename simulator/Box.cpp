#include "Box.h"
#include <algorithm>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <string.h>

void ParticleBox::initRandomly(double initialKineticEnergy, double initialGravitationalPotential) {
  double approxHeight = initialGravitationalPotential / (PARTICLE_MASS * GRAVITY);
  for (size_t i = 0; i < PARTICLES; i++) {
    double closestNeighbourDist = 0;
    while (closestNeighbourDist < 0.8 * LJ_SIGMA) {
      positions[i][0] = ((double)rand() / RAND_MAX) * 2 - 1;
      positions[i][1] = 1.0;
      closestNeighbourDist = LJ_SIGMA;
      for (size_t j = 0; j < i; j++)
        closestNeighbourDist = std::min(closestNeighbourDist, distanceBetween(i, j));
      std::cout << "." << std::flush;
    }

    std::cout << "Init particle at " << positions[i][0] << ", " << positions[i][1] << std::endl;
    velocities[i][0] = ((double)rand() / RAND_MAX) - 0.5;
    // velocities[i][0] = 0;
    velocities[i][1] = 0;
  }
}

void ParticleBox::f(ParticleVectors &accelerations) {
  for (size_t i = 0; i < PARTICLES; i++) {
    double force_x = 0, force_y = 0;
    for (size_t j = 0; j < PARTICLES; j++) {
      if (i == j)
        continue;
      double dx = positions[i][0] - positions[j][0], dy = positions[i][1] - positions[j][1];
      double r_squared = dx * dx + dy * dy;
      if (r_squared < LJ_CUT_DIST_SQ)
        r_squared = LJ_CUT_DIST_SQ;
      double r = sqrt(r_squared);
      force_x += dx * (pow(r, ALPHA - 1) - pow(r, BETA - 1));
      // force_y += dy * factor;
      // std::cout << "Distance_squared:" << r_squared << std::endl;
    }
    // std::cout << "Force: " << 24 * LJ_EPSILON * force_x << ", " << 24 * LJ_EPSILON * force_y - GRAVITY << std::endl;
    accelerations[i][0] = force_x / PARTICLE_MASS;
    accelerations[i][1] = 0;
  }
}

void ParticleBox::simulate(size_t timesteps) {
  double after_accelerations[PARTICLES][2];
  f(after_accelerations);
  for (size_t t = 0; t < timesteps; t++) {
    ParticleVectors before_accelerations = after_accelerations;
    memcpy(before_accelerations, after_accelerations, PARTICLES * 2 * sizeof(double));
    for (size_t i = 0; i < PARTICLES; i++) {
      positions[i][0] += TAU * velocities[i][0] + square(TAU) / 2 * before_accelerations[i][0];
      positions[i][1] += TAU * velocities[i][1] + square(TAU) / 2 * before_accelerations[i][1];
    }

    f(after_accelerations);
    double totalVelocity = 0;
    for (size_t i = 0; i < PARTICLES; i++) {
      velocities[i][0] += TAU / 2 * (before_accelerations[i][0] + after_accelerations[i][0]);
      velocities[i][1] += TAU / 2 * (before_accelerations[i][1] + after_accelerations[i][1]);
      totalVelocity += square(velocities[i][0]) + square(velocities[i][1]);
    }
    totalMeanVelocity += totalVelocity / PARTICLES;
    reflectParticles();
    // time += TIME_STEP
  }
}

void ParticleBox::reflectParticles() {
  for (size_t i = 0; i < PARTICLES; i++) {
    if (positions[i][0] < -1.0) {
      positions[i][0] = -1.0 - (1.0 + positions[i][0]); // assumes linear movement in this timestep
      velocities[i][0] = -velocities[i][0];
      // velocities[i][0] = 0;
    } else if (positions[i][0] > 1.0) {
      positions[i][0] = 1.0 - (positions[i][0] - 1.0); // assumes linear movement
      velocities[i][0] = -velocities[i][0];
      // velocities[i][0] = 0;
    }
    if (positions[i][1] < 0) {
      positions[i][1] = -positions[i][1]; // assumes linear movement
      velocities[i][1] = -velocities[i][1];
    }
  }
}

double ParticleBox::getKineticEnergy() {
  double energy = 0;
  for (size_t i = 0; i < PARTICLES; i++)
    energy += square(velocities[i][0]) + square(velocities[i][1]);
  return PARTICLE_MASS / 2 * energy;
}
double ParticleBox::getGravitationalPotential() {
  double totalHeight = 0;
  for (size_t i = 0; i < PARTICLES; i++)
    totalHeight += positions[i][1];
  return PARTICLE_MASS * GRAVITY * totalHeight;
}
double ParticleBox::getLJPotential() {
  double energy = 0;
  for (size_t i = 0; i < PARTICLES; i++) {
    for (size_t j = 0; j < PARTICLES; j++) {
      if (i == j)
        continue;
      double dx = positions[i][0] - positions[j][0], dy = positions[i][1] - positions[j][1];
      double r_squared = dx * dx + dy * dy;
      if (r_squared < LJ_CUT_DIST_SQ)
        r_squared = LJ_CUT_DIST_SQ;
      double r = sqrt(r_squared);
      energy += std::pow(r, ALPHA) / ALPHA - std::pow(r, BETA) / BETA;
      // std::cout << r_squared << ", " << energy << std::endl;
    }
  }
  return abs(energy);
}
double ParticleBox::getTotalEnergy() { return getKineticEnergy() + getGravitationalPotential() + getLJPotential(); }

void ParticleBox::computeHeightHistogram() {
  heightHist.min = -1;
  heightHist.max = 1;
  const double delta = heightHist.max - heightHist.min;
  std::fill(std::begin(heightHist.heights), std::end(heightHist.heights), 0);
  for (size_t i = 0; i < PARTICLES; i++) {
    size_t bin = floor((positions[i][0] - heightHist.min) / delta * HEIGHT_HISTOGRAM_BINS);
    if (bin >= HEIGHT_HISTOGRAM_BINS) // true for the last value
      bin = HEIGHT_HISTOGRAM_BINS - 1;
    heightHist.heights[bin]++;
  }
  heightHist.maxHeight = 0;
  for (size_t bin = 0; bin < VELOCITY_HISTOGRAM_BINS; bin++)
    heightHist.maxHeight = std::max(heightHist.maxHeight, heightHist.heights[bin]);
}

void ParticleBox::computeVelocityHistogram() {
  std::array<double, PARTICLES> values;
  for (size_t i = 0; i < PARTICLES; i++)
    values[i] = sqrt(square(velocities[i][0]) + square(velocities[i][1]));
  const auto [_min, _max] = std::minmax_element(values.begin(), values.end());
  velocityHist.min = *_min;
  velocityHist.max = *_max;
  const double delta = velocityHist.max - velocityHist.min;
  std::fill(std::begin(velocityHist.heights), std::end(velocityHist.heights), 0);
  for (size_t i = 0; i < PARTICLES; i++) {
    size_t bin = floor((values[i] - velocityHist.min) / delta * VELOCITY_HISTOGRAM_BINS);
    if (bin >= VELOCITY_HISTOGRAM_BINS)
      bin = VELOCITY_HISTOGRAM_BINS - 1;
    velocityHist.heights[bin]++;
  }
  velocityHist.maxHeight = 0;
  for (size_t bin = 0; bin < VELOCITY_HISTOGRAM_BINS; bin++)
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
}
