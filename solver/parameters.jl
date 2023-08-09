@kwdef struct AttractiveRepulsive
  alpha = 2.0  # attractive parameter
  beta = 1.0  # repulsive parameter
end

@kwdef struct MorsePotential
  C_att = 1.5
  l_att = 2.0
  C_rep = 1.0
  l_rep = 0.5
end

"""Evaluates the potential function. Multiple dispatch is not supported in the parameters."""
function potentialFunction(r; pot)
  if isa(pot, AttractiveRepulsive)
    return r^pot.alpha / pot.alpha - r^pot.beta / pot.beta
  elseif isa(pot, MorsePotential)
    return pot.C_rep * exp(-r / pot.l_rep) - pot.C_att * exp(-r / pot.l_att)
  else
    error("What is this potential?")
  end
end

"""The solver takes a few parameters"""
@kwdef struct Parameters
  d = 1  # dimension
  m = 1  # integer
  R0 = 0.8  # radius of the interval [-R, R]
  potential = AttractiveRepulsive()  # potential parameters
  M = 5  # number of basis elements to expand the general kernel in
end

function checkParameters(p::Parameters)
  if isa(p.potential, AttractiveRepulsive)
    @assert p.potential.alpha > p.potential.beta  # must be more attractive than repulsive
    @assert -p.d < p.potential.alpha < 2 + 2 * p.m - p.d
    @assert p.potential.beta > -p.d
    @assert p.m >= 0 && isinteger(p.m)
  end
end

defaultParams = Parameters()
known2dParams = Parameters(potential=AttractiveRepulsive(alpha=1.2, beta=0.1993), d=2)
knownAnalyticParams = Parameters(potential=AttractiveRepulsive(alpha=2.0001, beta=1.5), d=1)
morsePotiParams = Parameters(potential=MorsePotential(), d=1)

checkParameters(defaultParams)
checkParameters(known2dParams)
checkParameters(knownAnalyticParams)
checkParameters(morsePotiParams)
