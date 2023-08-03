@kwdef struct Parameters
  d = 1  # dimension
  m = 1  # integer
  alpha = 2.0001  # attractive parameter
  beta = 1.5  # repulsive parameter
  R0 = 0.8  # radius of the interval [-R, R]
  p = 1.0  # power parameter of the morse potential
  InteractionPotential = r -> exp(-r^p / p)  # actual interaction potential function
  M = 5  # number of basis elements to expand the function in
end

p = Parameters()
@assert -p.d < p.alpha < 2 + 2p.m - p.d
@assert p.beta > -p.d
@assert p.m >= 0 && isinteger(p.m)
