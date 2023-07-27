using ClassicalOrthogonalPolynomials, ContinuumArrays, Formatting, HypergeometricFunctions, SpecialFunctions
import ContinuumArrays: MappedWeightedBasisLayout, Map, WeightedBasisLayout

d = 2  # dimension
m = 1  # integer
alpha = 1.31  # attractive parameter
beta = 1.23  # repulsive parameter
R = 0.8372415  # radius of the interval [-R, R]
@assert -d < alpha < 2 + 2m - d
@assert beta > -d
@assert m >= 0 && isinteger(m)

"""Docstring for the function"""
function theorem216(r::Float64, n::Int64, beta::Float64=beta)::Float64
  # Explicit value of the integral from Theorem 2.16
  prefactor =
    pi^(d / 2) *
    gamma(1 + beta / 2) *
    gamma((beta + d) / 2) *
    gamma(m + n - (alpha + d) / 2 + 1) / (
      gamma(d / 2) *
      gamma(n + 1) *
      gamma(beta / 2 - n + 1) *
      gamma((beta - alpha) / 2 + m + n + 1)
    )
  integral_value =
    prefactor * HypergeometricFunctions._₂F₁.(
      n - beta / 2,
      -m - n + (alpha - beta) / 2,
      d / 2,
      abs.(r .^ 2),
    )
  # @show integral_value
  return integral_value
end

"""Docstring for the function"""
function recurrence(oldestValue, oldValue, r, n, beta=beta)
  # using Corollary 2.18
  c_a = -((-alpha + 2m + 4n) * (-alpha + 2m + 4n + 2) * (alpha + d - 2 * (m + n + 1))) /
        (2 * (n + 1) * (-alpha + beta + 2m + 2n + 2) * (-alpha + beta + d + 2m + 2n))
  c_b = -((-alpha + 2m + 4n) * (alpha + d - 2(m + n + 1)) * (d * (-alpha + 2 * beta + 2m + 2) - 2 * (2n - beta) * (-alpha + beta + 2m + 2n))) /
        (2 * (n + 1) * (-alpha + 2m + 4n - 2) * (-alpha + beta + 2m + 2n + 2) * (-alpha + beta + d + 2m + 2n))
  c_c = ((-beta + 2n - 2) * (beta + d - 2n) * (-alpha + 2m + 4n + 2) * (alpha + d - 2 * (m + n)) * (alpha + d - 2 * (m + n + 1))) /
        (4n * (n + 1) * (-alpha + 2m + 4n - 2) * (-alpha + beta + 2m + 2n + 2) * (-alpha + beta + d + 2m + 2n))
  return (c_a * r^2 + c_b) * oldValue + c_c * oldestValue
end

"""Docstring for the function"""
function totalEnergy(solution::Vector{Float64}, r=0.0)::Float64
  # more details in section 3.2
  attractive, repulsive = 0.0, 0.0
  for k in eachindex(solution)
    attractive += solution[k] * theorem216(r, k, alpha)
    repulsive += solution[k] * theorem216(r, k, beta)
  end
  E = (R^(alpha + d) / alpha) * attractive - (R^(beta + d) / beta) * repulsive
  return E
end

"""Docstring for the function"""
function totalMass(solution::Vector{Float64})::Float64
  # using Lemma 2.20
  return pi^(d / 2) * gamma(B.a + 1) / gamma(B.a + d / 2 + 1) * solution[1]
end

# These definitions allow the use of the radially shifted Jacobi bases
struct QuadraticMap{T} <: Map{T} end
struct InvQuadraticMap{T} <: Map{T} end
QuadraticMap() = QuadraticMap{Float64}()
InvQuadraticMap() = InvQuadraticMap{Float64}()
Base.getindex(::QuadraticMap, r::Number) = 2r^2 - 1
Base.axes(::QuadraticMap{T}) where {T} = (Inclusion(0 .. 1),)
Base.axes(::InvQuadraticMap{T}) where {T} = (Inclusion(-1 .. 1),)
Base.getindex(d::InvQuadraticMap, x::Number) = sqrt((x + 1) / 2)
ContinuumArrays.invmap(::QuadraticMap{T}) where {T} = InvQuadraticMap{T}()
ContinuumArrays.invmap(::InvQuadraticMap{T}) where {T} = QuadraticMap{T}()
Base.getindex(d::QuadraticMap, x::Inclusion) = d
Base.getindex(d::InvQuadraticMap, x::Inclusion) = d

map = QuadraticMap();
imap = InvQuadraticMap();

# represent the basis P_n^(a,b)(2r^2-1)
# apparently this is wrong: B = Jacobi(m - (alpha + d) / 2, (d - 2) / 2); the following is correct
B = Jacobi(m - (beta + d) / 2, (d - 2) / 2);
P = B[map, :];
x = axes(B, 1)
r = axes(P, 1)

@assert x.domain == (-1 .. 1)  # Chebyshev
@assert r.domain == (0 .. 1)  # Radial

"""Docstring for the function"""
function constructOperator(N::Int64, beta::Float64)::Array{Float64,2}
  Matrix = zeros(N, N)
  for n in 0:N-1
    Function = theorem216.(r, n, beta)
    ExpansionCoeffs = P[:, 1:N] \ Function  # expands the function in the P basis
    Matrix[:, n+1] .= ExpansionCoeffs
  end
  return Matrix
end

"""Docstring for the function"""
function recursivelyConstructOperatorWithReprojection(N::Int64, beta::Float64)::Array{Float64,2}
  Matrix = zeros(N, N)
  OldestFunction = theorem216.(r, 0, beta)
  OldFunction = theorem216.(r, 1, beta)

  Matrix[:, 1] = P[:, 1:N] \ OldestFunction
  if N < 2
    return Matrix
  end

  Matrix[:, 2] = P[:, 1:N] \ OldFunction
  if N < 3
    return Matrix
  end

  for remainingColumn in 3:N
    Function = recurrence.(OldestFunction, OldFunction, r, remainingColumn - 2, beta)
    Matrix[:, remainingColumn] = P[:, 1:N] \ Function
    OldestFunction = OldFunction
    OldFunction = Function
  end
  return Matrix
end

"""Docstring for the function"""
function solve(N::Int64)::Vector{Float64}
  AttractiveMatrix = constructOperator(N, alpha)
  RepulsiveMatrix = constructOperator(N, beta)
  BigMatrix = (R^(alpha) / alpha) * AttractiveMatrix - (R^(beta) / beta) * RepulsiveMatrix
  BigRHS = zeros(N)
  BigRHS[1] = 1

  BigSolution = BigMatrix \ BigRHS
  return BigSolution / totalMass(BigSolution)
end

"""In possession of a solution, evaluates the measure (function) at given values of x."""
function rho(x_vec, solution::Vector{Float64})
  return (1 .- x_vec .^ 2) .^ B.a .* vec(sum(solution .* P[abs.(x_vec), 1:length(solution)]', dims=1))
end

println("Have fun solving!")
