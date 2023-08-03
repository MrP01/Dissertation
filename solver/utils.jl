module Utils
using ContinuumArrays
import ClassicalOrthogonalPolynomials: Jacobi
import HypergeometricFunctions
import ContinuumArrays: Map
import SpecialFunctions: gamma

include("./parameters.jl")

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
function createBasis(p::Parameters=p)
  # TODO: which is it? alpha or beta?
  B = Jacobi(p.m - (p.alpha + p.d) / 2, (p.d - 2) / 2)
  P = B[Utils.qmap, :]
  return B, P
end

struct SolutionEnvironment
  p::Parameters
  B::Jacobi  # Jacobi Basis
  P::ContinuumArrays.QuasiArrays.SubQuasiArray # with argument mapping
end

"""Creates a fresh environment based on the parameters."""
function createEnvironment(p::Parameters=p)
  B, P = createBasis()
  return SolutionEnvironment(p, B, P)
end

defaultEnv = createEnvironment(p)

"""Docstring for the function"""
function theorem216(r::Real, n::Int64, beta::Float64=p.beta)::BigFloat
  # Explicit value of the integral from Theorem 2.16
  prefactor =
    pi^(p.d / 2) *
    gamma(1 + beta / 2) *
    gamma((beta + p.d) / 2) *
    gamma(p.m + n - (p.alpha + p.d) / 2 + 1) / (
      gamma(p.d / 2) *
      gamma(n + 1) *
      gamma(beta / 2 - n + 1) *
      gamma((beta - p.alpha) / 2 + p.m + n + 1)
    )
  integral_value =
    prefactor * HypergeometricFunctions._₂F₁.(
      big(n - beta / 2),
      big(-p.m - n + (p.alpha - beta) / 2),
      big(p.d / 2),
      big(abs.(r .^ 2)),
    )
  # @show integral_value
  return integral_value
end

"""Docstring for the function"""
function recurrence(oldestValue, oldValue, r, n, beta=p.beta)
  # using Corollary 2.18
  m = p.m
  c_a = -((-p.alpha + 2m + 4n) * (-p.alpha + 2m + 4n + 2) * (p.alpha + p.d - 2 * (p.m + n + 1))) /
        (2 * (n + 1) * (-p.alpha + beta + 2m + 2n + 2) * (-p.alpha + beta + p.d + 2m + 2n))
  c_b = -((-p.alpha + 2m + 4n) * (p.alpha + p.d - 2(p.m + n + 1)) * (p.d * (-p.alpha + 2 * beta + 2m + 2) - 2 * (2n - beta) * (-p.alpha + beta + 2m + 2n))) /
        (2 * (n + 1) * (-p.alpha + 2m + 4n - 2) * (-p.alpha + beta + 2m + 2n + 2) * (-p.alpha + beta + p.d + 2m + 2n))
  c_c = ((-beta + 2n - 2) * (beta + p.d - 2n) * (-p.alpha + 2m + 4n + 2) * (p.alpha + p.d - 2 * (p.m + n)) * (p.alpha + p.d - 2 * (p.m + n + 1))) /
        (4n * (n + 1) * (-p.alpha + 2m + 4n - 2) * (-p.alpha + beta + 2m + 2n + 2) * (-p.alpha + beta + p.d + 2m + 2n))
  return (c_a * r^2 + c_b) * oldValue + c_c * oldestValue
end

"""In possession of a solution, evaluates the measure (function) at given values of x."""
function rho(x_vec, solution::Vector{BigFloat}, env::SolutionEnvironment=defaultEnv)
  return (1 .- x_vec .^ 2) .^ env.B.a .* vec(sum(solution .* env.P[abs.(x_vec), 1:length(solution)]', dims=1))
end

"""Docstring for the function"""
function totalEnergy(solution::Vector{BigFloat}, R=p.R0::Float64, r=0.0::Float64)::BigFloat
  # more details in section 3.2
  attractive, repulsive = 0.0, 0.0
  for k in eachindex(solution)
    attractive += solution[k] * theorem216(r, k, p.alpha)
    repulsive += solution[k] * theorem216(r, k, p.beta)
  end
  E = (R^(p.alpha + p.d) / p.alpha) * attractive - (R^(p.beta + p.d) / p.beta) * repulsive
  return E
end

"""Docstring for the function"""
function totalMass(solution::Vector{BigFloat}, env::SolutionEnvironment=defaultEnv)::BigFloat
  # using Lemma 2.20
  return pi^(p.d / 2) * gamma(env.B.a + 1) / gamma(env.B.a + p.d / 2 + 1) * solution[1]
end

"""Sets small values in a matrix to zero. Improves accuracy of the solutions by a tiny bit!"""
function zeroOutTinyValues!(M::Array{BigFloat,2})
  M[abs.(M).<big"1e-12"] .= 0
end
end
