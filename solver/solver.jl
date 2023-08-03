module Solver
using ClassicalOrthogonalPolynomials, ContinuumArrays
import ContinuumArrays: MappedWeightedBasisLayout, WeightedBasisLayout
import LinearAlgebra: cond
import SpecialFunctions: gamma
import Optim

include("./parameters.jl")
import ..Utils

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
function constructOperator(N::Int64, beta::Float64, env::SolutionEnvironment=defaultEnv)::Array{BigFloat,2}
  Mat = zeros(BigFloat, N, N)
  r = axes(env.P, 1)
  for n in 0:N-1
    Function = Utils.theorem216.(r, n, beta)
    ExpansionCoeffs = env.P[:, 1:N] \ Function  # expands the function in the P basis
    Mat[:, n+1] .= ExpansionCoeffs
  end
  Utils.zeroOutTinyValues!(Mat)
  return Mat
end

"""Recursively constructs with reprojection, terrible because the types keep on nesting inside of one another."""
function recursivelyConstructOperatorWithReprojection(N::Int64, beta::Float64, env::SolutionEnvironment=defaultEnv)::Array{BigFloat,2}
  Mat = zeros(BigFloat, N, N)
  r = axes(env.P, 1)
  OldestFunction = Utils.theorem216.(r, 0, beta)
  OldFunction = Utils.theorem216.(r, 1, beta)

  Mat[:, 1] = env.P[:, 1:N] \ OldestFunction
  if N < 2
    return Mat
  end

  Mat[:, 2] = env.P[:, 1:N] \ OldFunction
  if N < 3
    return Mat
  end

  for remainingColumn in 3:N
    Function = Utils.recurrence.(OldestFunction, OldFunction, r, remainingColumn - 2, beta)
    Mat[:, remainingColumn] = env.P[:, 1:N] \ Function
    OldestFunction = OldFunction
    OldFunction = Function
  end
  Utils.zeroOutTinyValues!(Mat)
  return Mat
end

"""Docstring for the function"""
function solve(N::Int64, R=p.R0::Float64, env::SolutionEnvironment=defaultEnv)::Vector{BigFloat}
  AttractiveMatrix = constructOperator(N, p.alpha, env)
  RepulsiveMatrix = constructOperator(N, p.beta, env)
  @show cond(convert(Matrix{Float64}, AttractiveMatrix))
  @show cond(convert(Matrix{Float64}, RepulsiveMatrix))
  BigMatrix = (R^p.alpha / p.alpha) * AttractiveMatrix - (R^p.beta / p.beta) * RepulsiveMatrix
  @show cond(convert(Matrix{Float64}, BigMatrix))
  BigRHS = zeros(N)
  BigRHS[1] = 1

  BigSolution = BigMatrix \ BigRHS
  return BigSolution / totalMass(BigSolution)
end

function outerOptimisation(N::Int64=20, env::SolutionEnvironment=defaultEnv)
  F(R) = totalEnergy(solve(N, R))
  f(x) = F(x[1])  # because optimize() only accepts vector inputs
  solution = Optim.optimize(f, [R0])
  return solution
end

"""In possession of a solution, evaluates the measure (function) at given values of x."""
function rho(x_vec, solution::Vector{BigFloat}, env::SolutionEnvironment=defaultEnv)
  return (1 .- x_vec .^ 2) .^ B.a .* vec(sum(solution .* env.P[abs.(x_vec), 1:length(solution)]', dims=1))
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

println("Have fun solving!")
end
