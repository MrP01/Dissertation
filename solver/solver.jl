module Solver
import ContinuumArrays: MappedWeightedBasisLayout, WeightedBasisLayout
import LinearAlgebra: cond
import SpecialFunctions: gamma
import LRUCache
import Optim

import ..Utils
import ..Utils: SolutionEnvironment, defaultEnv, Parameters, defaultParams

OpMem = LRUCache.LRU{Tuple{Int64,Float64,SolutionEnvironment},Matrix{BigFloat}}(maxsize=20)
"""Creates an operator given alpha and beta={alpha, beta}. Caches it."""
function constructOperator(N::Int64, beta::Float64, env::SolutionEnvironment=defaultEnv)::Matrix{BigFloat}
  get!(OpMem, (N, beta, env)) do
    Mat = zeros(BigFloat, N, N)
    r = axes(env.P, 1)
    for n in 0:N-1
      Function = Utils.theorem216.(r, n, beta, env.p)
      ExpansionCoeffs = env.P[:, 1:N] \ Function  # expands the function in the P basis
      Mat[:, n+1] .= ExpansionCoeffs
    end
    Utils.zeroOutTinyValues!(Mat)
    Mat
  end
end

"""Recursively constructs with reprojection, terrible because the types keep on nesting inside of one another."""
function recursivelyConstructOperatorWithReprojection(N::Int64, beta::Float64, env::SolutionEnvironment=defaultEnv)::Matrix{BigFloat}
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

function constructFullOperator(N::Int64, R::Float64, env::SolutionEnvironment=defaultEnv)::Matrix{BigFloat}
  p::Parameters = env.p
  AttractiveMatrix = constructOperator(N, p.alpha, env)
  RepulsiveMatrix = constructOperator(N, p.beta, env)
  # @show cond(convert(Matrix{Float64}, AttractiveMatrix))
  # @show cond(convert(Matrix{Float64}, RepulsiveMatrix))
  return (R^p.alpha / p.alpha) * AttractiveMatrix - (R^p.beta / p.beta) * RepulsiveMatrix
end

"""Docstring for the function"""
function solve(N::Int64, R=defaultParams.R0::Float64, env::SolutionEnvironment=defaultEnv)::Vector{BigFloat}
  BigMatrix = constructFullOperator(N, R, env)
  # @show cond(convert(Matrix{Float64}, BigMatrix))
  BigRHS = zeros(N)
  BigRHS[1] = 1.0

  BigSolution = BigMatrix \ BigRHS
  return BigSolution / Utils.totalMass(BigSolution, env)
end

"""Docstring for the function"""
function outerOptimisation(N::Int64=20, env::SolutionEnvironment=defaultEnv)
  F(R) = Utils.totalEnergy(solve(N, R, env), R, 0.0, env)
  f(x) = F(x[1])  # because optimize() only accepts vector inputs
  solution = Optim.optimize(f, [R0])
  return solution
end

println("Have fun solving!")
end
