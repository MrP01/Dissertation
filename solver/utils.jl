module Utils
using ContinuumArrays
import ClassicalOrthogonalPolynomials: Jacobi
import HypergeometricFunctions
import ContinuumArrays: Map
import SpecialFunctions: gamma
import ..Params

PARAMETER_TO_FIND = 2.5  # probably, most dominant term in the monomial expansion, so maximum(abs(monomial))

# These definitions allow the use of the radially shifted Jacobi bases
struct QuadraticMap{T} <: Map{T} end
struct InvQuadraticMap{T} <: Map{T} end
QuadraticMap() = QuadraticMap{Float64}()
InvQuadraticMap() = InvQuadraticMap{Float64}()
Base.getindex(::QuadraticMap, r::Number) = 2r^2 - 1
Base.axes(::QuadraticMap{T}) where {T} = (Inclusion(0 .. 1),)
Base.axes(::InvQuadraticMap{T}) where {T} = (Inclusion(-1 .. 1),)
Base.getindex(map::InvQuadraticMap, x::Number) = sqrt((x + 1) / 2)
ContinuumArrays.invmap(::QuadraticMap{T}) where {T} = InvQuadraticMap{T}()
ContinuumArrays.invmap(::InvQuadraticMap{T}) where {T} = QuadraticMap{T}()
Base.getindex(map::QuadraticMap, x::Inclusion) = map
Base.getindex(map::InvQuadraticMap, x::Inclusion) = map

qmap = QuadraticMap()
iqmap = InvQuadraticMap()

"""Represent the basis P_n^(a,b)(2r^2-1)"""
function createBasis(p::Params.Parameters)
  # TODO: which is it? alpha or beta?
  if isa(p.potential, Params.AttractiveRepulsive)
    alpha = p.potential.alpha
  elseif isa(p.potential, Params.MorsePotential)
    alpha = PARAMETER_TO_FIND  # TODO: what should we put here?
  else
    error("Unkown potential")
  end
  B = Jacobi(p.m - (alpha + p.d) / 2, (p.d - 2) / 2)
  P = B[Utils.qmap, :]
  return B, P
end

"""Docstring for the function. M: number of basis elements to expand the general kernel in."""
function basisConversionMatrix(P, M=5)
  r = axes(P, 1)
  return mapreduce(permutedims, hcat, [P[:, 1:M] \ r .^ k for k in 0:M-1]')
end

"""Docstring for the function. M: number of basis elements to expand the general kernel in."""
function expandKernelInMonomials(potential, P, M=5)
  r = axes(P, 1)
  BasisConversionMat = basisConversionMatrix(P, M)
  InteractionCoeffs = convert(Vector{Float64}, P[:, 1:M] \ Params.potentialFunction.(r; pot=potential))  # in Jacobi basis
  MonomialInteractionCoeffs = BasisConversionMat \ InteractionCoeffs  # in monomial basis
  return MonomialInteractionCoeffs
end

struct SolutionEnvironment
  p::Params.Parameters
  B::Jacobi  # Jacobi Basis
  P::ContinuumArrays.QuasiArrays.SubQuasiArray # with argument mapping
  monomial::Vector{Float64}
end

"""Creates a fresh environment based on the Params.Parameters."""
function createEnvironment(p::Params.Parameters, M=5)::SolutionEnvironment
  B, P = createBasis(p)
  monomial = []
  if ~isa(p.potential, Params.AttractiveRepulsive)
    monomial = expandKernelInMonomials(p.potential, P, M)
  end
  return SolutionEnvironment(p, B, P, monomial)
end

defaultEnv::SolutionEnvironment = createEnvironment(Params.defaultParams)

"""Docstring for the function"""
function theorem216(r::Real; n::Int64, beta::Float64, p::Params.Parameters)::BigFloat
  # Explicit value of the integral from Theorem 2.16
  if isa(p.potential, Params.AttractiveRepulsive)
    alpha = p.potential.alpha
  elseif isa(p.potential, Params.MorsePotential)
    alpha = PARAMETER_TO_FIND  # TODO: what should we put here?
  else
    error("Unkown potential")
  end
  prefactor =
    pi^(p.d / 2) *
    gamma(1 + beta / 2) *
    gamma((beta + p.d) / 2) *
    gamma(p.m + n - (alpha + p.d) / 2 + 1) / (
      gamma(p.d / 2) *
      gamma(n + 1) *
      gamma(beta / 2 - n + 1) *
      gamma((beta - alpha) / 2 + p.m + n + 1)
    )
  integral_value =
    prefactor * HypergeometricFunctions._₂F₁.(
      big(n - beta / 2),
      big(-p.m - n + (alpha - beta) / 2),
      big(p.d / 2),
      big(abs.(r .^ 2)),
    )
  # @show integral_value
  return integral_value
end

"""In possession of a solution, evaluates the measure (function) at given values of x."""
function rho(x_vec, solution::Vector{BigFloat}, env::SolutionEnvironment)
  return (1 .- x_vec .^ 2) .^ env.B.a .* vec(sum(solution .* env.P[abs.(x_vec), 1:length(solution)]', dims=1))
end

"""Docstring for the function"""
function totalEnergy(solution::Vector{BigFloat}, R::Float64, r::Union{Float64,AbstractVector{Float64}}, env::SolutionEnvironment)
  # more details in section 3.2
  if isa(env.p.potential, Params.AttractiveRepulsive)
    alpha, beta = env.p.potential.alpha, env.p.potential.beta
    attractive, repulsive = zero(r), zero(r)
    for k in eachindex(solution)
      attractive += solution[k] * Float64.(theorem216.(r; n=k - 1, beta=alpha, p=env.p))
      repulsive += solution[k] * Float64.(theorem216.(r; n=k - 1, beta=beta, p=env.p))
    end
    E = (R^alpha / alpha) * attractive - (R^beta / beta) * repulsive
  else
    E = zero(r)
    for k in eachindex(solution)
      for index in eachindex(env.monomial)
        # index starts from 1!
        power, coefficient = index - 1, env.monomial[index]
        E += coefficient * R^power * solution[k] * Float64.(theorem216.(r; n=k - 1, beta=float(power) + 0.0001, p=env.p))
      end
    end
  end
  return E
end

"""Docstring for the function"""
function totalMass(solution::Vector{BigFloat}, env::SolutionEnvironment)::BigFloat
  # using Lemma 2.20
  p::Params.Parameters = env.p
  return pi^(p.d / 2) * gamma(env.B.a + 1) / gamma(env.B.a + p.d / 2 + 1) * solution[1]
end

"""Sets small values in a matrix to zero. Improves accuracy of the solutions by a tiny bit!"""
function zeroOutTinyValues!(M::Matrix{BigFloat})
  M[abs.(M).<big"1e-12"] .= 0
end
end
