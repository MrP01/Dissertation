"""The solver takes a few parameters"""
@kwdef struct Parameters
  d = 2  # dimension
  m = 1  # integer
  alpha = 1.2  # attractive parameter
  beta = 0.1993  # repulsive parameter
  R0 = 0.8  # radius of the interval [-R, R]
  p = 1.0  # power parameter of the morse potential
  InteractionPotential = r -> exp(-r^p / p)  # actual interaction potential function
  M = 5  # number of basis elements to expand the function in
end

defaultParams = Parameters()
@assert -defaultParams.d < defaultParams.alpha < 2 + 2 * defaultParams.m - defaultParams.d
@assert defaultParams.beta > -defaultParams.d
@assert defaultParams.m >= 0 && isinteger(defaultParams.m)
