module Solver
import ContinuumArrays: MappedWeightedBasisLayout, WeightedBasisLayout
import LinearAlgebra: cond
import SpecialFunctions: gamma
import Optim

include("./parameters.jl")
import ..Utils
import ..Utils: SolutionEnvironment, defaultEnv

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
  return BigSolution / Utils.totalMass(BigSolution)
end

function outerOptimisation(N::Int64=20, env::SolutionEnvironment=defaultEnv)
  F(R) = Utils.totalEnergy(solve(N, R))
  f(x) = F(x[1])  # because optimize() only accepts vector inputs
  solution = Optim.optimize(f, [R0])
  return solution
end

println("Have fun solving!")
end
