module Params
@kwdef struct AttractiveRepulsive
  alpha = 1.8  # attractive parameter
  beta = 1.2  # repulsive parameter
end

@kwdef struct MorsePotential
  C_att = 1.5
  l_att = 2.0
  C_rep = 1.0
  l_rep = 0.5
end

@kwdef struct MixedPotential
  morseC = 1.0
  morsel = 0.5
  attrepPower = 1.8
end

@kwdef struct AbsoluteValuePotential
end

@kwdef struct QuadraticSelfPropulsion
  selfPropulsion = 0.0
  frictionCoeff = 0.5
end

"""Evaluates the potential function. Multiple dispatch is not supported in the parameters."""
function potentialFunction(r; pot)
  if isa(pot, AttractiveRepulsive)
    return r^pot.alpha / pot.alpha - r^pot.beta / pot.beta
  elseif isa(pot, MorsePotential)
    return pot.C_rep * exp(-r / pot.l_rep) - pot.C_att * exp(-r / pot.l_att)
  elseif isa(pot, MixedPotential)
    return pot.morseC * exp(-r / pot.morsel) + r^pot.attrepPower / pot.attrepPower
  else
    error("What is this potential?")
  end
end

function potentialParamsToLatex(pot, rounded=false)
  if isa(pot, AttractiveRepulsive)
    alpha, beta = pot.alpha, pot.beta
    if rounded
      alpha, beta = round(pot.alpha, digits=2), round(pot.beta, digits=2)
    end
    return "(\\alpha, \\beta) = ($(alpha), $(beta))"
  elseif isa(pot, MorsePotential)
    return "(C_a, l_a, C_r, l_r) = ($(pot.C_att), $(pot.l_att), $(pot.C_rep), $(pot.l_rep))"
  elseif isa(pot, MixedPotential)
    return "(C, l, a) = ($(pot.morseC), $(pot.morsel), $(pot.attrepPower))"
  elseif isa(pot, AbsoluteValuePotential)
    return "K(r) = |1-r|"
  else
    error("What is this potential?")
  end
end

"""The solver takes a few parameters"""
@kwdef struct Parameters
  d = 1  # dimension
  m = 1  # integer
  R0 = 0.8  # radius of the interval [-R, R]
  s0 = 1e-12  # regularisation parameter
  potential = AttractiveRepulsive()  # potential parameters
  friction = QuadraticSelfPropulsion()  # friction
  name::String = "attrep"
end

function checkParameters(p::Parameters)
  if isa(p.potential, AttractiveRepulsive)
    @assert p.potential.alpha > p.potential.beta  # must be more attractive than repulsive
    @assert -p.d < p.potential.alpha < 2 + 2 * p.m - p.d
    @assert p.potential.beta > -p.d
    @assert p.m >= 0 && isinteger(p.m)
  end
end

function parametersToDict(p::Parameters)
  return Dict(s => getfield(p, s) for s in fieldnames(typeof(p)))
end
function dictToParameters(d::Dict)
  return Parameters(; d...)
end
function copyWithChanges(p::Parameters; kwargs...)
  d = parametersToDict(p)
  return dictToParameters(merge(d, kwargs))
end

defaultParams = Parameters()
known2dParams = Parameters(potential=AttractiveRepulsive(alpha=1.2, beta=0.1993), d=2, R0=1.5, name="known-2d")
knownAnalyticParams = Parameters(potential=AttractiveRepulsive(alpha=2.0, beta=1.5), d=1, name="known-analytic")
morsePotiParams = Parameters(potential=MorsePotential(), d=1, s0=1e-5, name="morse")
morsePotiSwarming2d = Parameters(potential=MorsePotential(), friction=QuadraticSelfPropulsion(selfPropulsion=1.6), d=2, m=2, s0=1e-5, name="morse-2d")
morsePotiSwarming3d = Parameters(potential=MorsePotential(), friction=QuadraticSelfPropulsion(selfPropulsion=1.6), d=3, m=2, s0=1e-5, name="morse-3d")
voidParams2d = Parameters(d=2, m=2, potential=AttractiveRepulsive(alpha=3.5, beta=1.6), name="void-2d")  # found in meeting with Timon
bumpParams = Parameters(d=1, m=0, potential=AttractiveRepulsive(alpha=0.912, beta=0.881), R0=1.4, name="bump")  # 2020-power-law, fig. 11
mixedParams1d = Parameters(d=1, m=0, potential=MixedPotential(), friction=QuadraticSelfPropulsion(selfPropulsion=1.6), name="mixed-1d")
mixedParams = Parameters(d=2, m=0, potential=MixedPotential(), friction=QuadraticSelfPropulsion(selfPropulsion=1.6), name="mixed-2d")
gyroscope3dParams = Parameters(d=3, potential=AbsoluteValuePotential(), friction=QuadraticSelfPropulsion(0.5, 0.5), name="gyroscope-3d")

checkParameters(defaultParams)
checkParameters(known2dParams)
checkParameters(knownAnalyticParams)
checkParameters(morsePotiParams)
checkParameters(morsePotiSwarming2d)
checkParameters(morsePotiSwarming3d)
checkParameters(voidParams2d)
checkParameters(bumpParams)
checkParameters(mixedParams)
checkParameters(mixedParams1d)
end
