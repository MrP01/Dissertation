"""The solver takes a few parameters"""
@kwdef struct Parameters
  d = 1  # dimension
  m = 1  # integer
  alpha = 1.2  # attractive parameter
  beta = 0.1993  # repulsive parameter
  R0 = 0.8  # radius of the interval [-R, R]
  p = 1.0  # power parameter of the morse potential
  InteractionPotential = r -> exp(-r^p / p)  # actual interaction potential function
  M = 5  # number of basis elements to expand the function in
end

function checkParameters(p::Parameters)
  @assert p.alpha > p.beta  # must be more attractive than repulsive
  @assert -p.d < p.alpha < 2 + 2 * p.m - p.d
  @assert p.beta > -p.d
  @assert p.m >= 0 && isinteger(p.m)
end

defaultParams = Parameters()
knownAnalyticParams = Parameters(d=1, alpha=2.0001, beta=1.5)

checkParameters(defaultParams)
checkParameters(knownAnalyticParams)
