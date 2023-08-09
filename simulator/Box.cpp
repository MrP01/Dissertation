#include "Box.h"
#include <algorithm>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <string.h>

double ParticleBox::friction(double v) { return p.selfPropulsion - p.friction * v * v; }

void ParticleBox::initRandomly() {
  for (size_t i = 0; i < PARTICLES; i++) {
    double closestNeighbourDist = 0;
    while (closestNeighbourDist < 0.1 / PARTICLES) {
      for (size_t d = 0; d < DIMENSION; d++)
        positions[i][d] = ((double)rand() / RAND_MAX - 0.5) * p.initWindowLength;
      closestNeighbourDist = 0.1 / PARTICLES;
      for (size_t j = 0; j < i; j++)
        closestNeighbourDist = std::min(closestNeighbourDist, distanceBetween(i, j));
      std::cout << "." << std::flush;
    }

    // std::cout << "Init particle at " << positions[i][0] << std::endl;
    for (size_t d = 0; d < DIMENSION; d++)
      velocities[i][d] = (((double)rand() / RAND_MAX) - 0.5);
  }
}

void ParticleBox::f(ParticleVectors &accelerations) {
  for (size_t i = 0; i < PARTICLES; i++) {
    double forces[DIMENSION] = {0.0};
    for (size_t j = 0; j < PARTICLES; j++) {
      if (i == j)
        continue;
      double r = distanceBetween(i, j);
      if (r < LJ_CUTOFF_DISTANCE)
        r = LJ_CUTOFF_DISTANCE;
      for (size_t d = 0; d < DIMENSION; d++)
        forces[d] += (positions[j][d] - positions[i][d]) / r * interaction->force(p.boxScaling * r);
    }
    // std::cout << "force " << forces[0] << std::endl;
    double v = totalVelocity(i); // is positive
    for (size_t d = 0; d < DIMENSION; d++)
      accelerations[i][d] = (forces[d] / PARTICLE_MASS + friction(p.boxScaling * v) * velocities[i][d]);
  }
}

void ParticleBox::simulate(size_t timesteps, bool dot) {
  double afterAccelerations[PARTICLES][DIMENSION];
  f(afterAccelerations);
  for (size_t t = 0; t < timesteps; t++) {
    ParticleVectors beforeAccelerations = afterAccelerations;
    memcpy(beforeAccelerations, afterAccelerations, PARTICLES * DIMENSION * sizeof(double));
    for (size_t i = 0; i < PARTICLES; i++) {
      for (size_t d = 0; d < DIMENSION; d++)
        positions[i][d] += p.tau * velocities[i][d] + square(p.tau) / 2 * beforeAccelerations[i][d];
    }

    f(afterAccelerations);
    for (size_t i = 0; i < PARTICLES; i++) {
      for (size_t d = 0; d < DIMENSION; d++) {
        velocities[i][d] += (p.tau / 2 * (beforeAccelerations[i][d] + afterAccelerations[i][d])) / p.boxScaling;
      }
    }
    reflectParticles();
    if (dot && t % 5 == 0)
      std::cout << "." << std::flush;
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
  return PARTICLE_MASS / 2 * energy;
}
double ParticleBox::getLJPotential() {
  double energy = 0;
  for (size_t i = 0; i < PARTICLES; i++) {
    for (size_t j = 0; j < PARTICLES; j++) {
      if (i == j)
        continue;
      double r = distanceBetween(i, j);
      if (r < LJ_CUTOFF_DISTANCE)
        r = LJ_CUTOFF_DISTANCE;
      energy -= interaction->potential(p.boxScaling * r);
      // std::cout << r << ", " << energy << std::endl;
    }
  }
  return energy;
}
double ParticleBox::getTotalEnergy() { return getKineticEnergy() + getLJPotential(); }

void ParticleBox::computeRadiusHistogram() {
  radiusHist.min = 0;
  radiusHist.max = sqrt(DIMENSION);
  const double delta = radiusHist.max - radiusHist.min;
  std::fill(std::begin(radiusHist.heights), std::end(radiusHist.heights), 0);
  double center[DIMENSION] = {0.0};
  for (size_t d = 0; d < DIMENSION; d++) {
    for (size_t i = 0; i < PARTICLES; i++)
      center[d] += positions[i][d];
    center[d] /= PARTICLES;
  }

  for (size_t i = 0; i < PARTICLES; i++) {
    double r_squared = 0;
    for (size_t d = 0; d < DIMENSION; d++)
      r_squared += square(positions[i][d] - center[d]);
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

  averagedRadiusHistogram.min = radiusHist.min;
  averagedRadiusHistogram.max = radiusHist.max;
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
  std::array<double, PARTICLES> values = {0};
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
  for (size_t i = 0; i < PARTICLES; i++) {
    for (size_t d = 0; d < DIMENSION - 1; d++)
      positionsCsv << positions[i][d] << ", ";
    positionsCsv << positions[i][DIMENSION - 1] << "\n";
  }
  positionsCsv.close();
  std::ofstream velocitiesCsv("/tmp/velocities.csv");
  for (size_t i = 0; i < PARTICLES; i++) {
    for (size_t d = 0; d < DIMENSION - 1; d++)
      velocitiesCsv << velocities[i][d] << ", ";
    velocitiesCsv << velocities[i][DIMENSION - 1] << "\n";
  }
  velocitiesCsv.close();
  std::ofstream positionHistCsv("/tmp/position-histogram.csv");
  for (size_t i = 0; i < RADIAL_HISTOGRAM_BINS; i++)
    positionHistCsv << averagedRadiusHistogram.heights[i] << "\n";
  positionHistCsv.close();
}

int ParticleBox::setupFromArgv(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: ./experiments (morse|attrep) dimension [iterations] [C_a, C_r, l_a, l_b | alpha, beta]"
              << std::endl;
    return 1;
  }

  char *potentialType = argv[1];

  size_t dimension = atol(argv[2]);
  if (dimension != DIMENSION) {
    std::cerr << "DIMENSION SHOULD BE " << DIMENSION << std::endl;
    std::cerr << "(Otherwise, have to recompile or use a different binary) " << DIMENSION << std::endl;
    return 1;
  }

  if (strcmp(potentialType, "attrep") == 0) {
    std::cout << "Using attractive-repulsive interaction potential" << std::endl;
    auto poti = new AttractiveRepulsive();
    if (argc >= 6) {
      poti->alpha = atof(argv[4]);
      poti->beta = atof(argv[5]);
    }
    std::cout << poti->alpha << ", " << poti->beta << std::endl;
    interaction = poti;
  } else if (strcmp(potentialType, "morse") == 0) {
    std::cout << "Using Morse interaction potential" << std::endl;
    auto poti = new MorsePotential();
    if (argc >= 8) {
      poti->C_att = atof(argv[4]);
      poti->C_rep = atof(argv[5]);
      poti->l_att = atof(argv[6]);
      poti->l_rep = atof(argv[7]);
    }
    interaction = poti;
  } else {
    std::cerr << "Unknown potential. Choices: attrep, morse." << std::endl;
    return 1;
  }
  return 0;
}
