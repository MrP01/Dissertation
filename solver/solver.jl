include("./utils.jl")
include("./attractiverepulsive.jl")
include("./generalkernel.jl")

module Solver
import ..Params
import ..Utils
import ..AttractiveRepulsiveSolver
import ..GeneralKernelSolver
import LinearAlgebra

function constructOperatorFromEnv(N::Int64, R::Float64, env::Utils.SolutionEnvironment)
  if isa(env.p.potential, Params.AttractiveRepulsive)
    BigMatrix = AttractiveRepulsiveSolver.constructFullOperator(N, R, env)
  else
    BigMatrix = GeneralKernelSolver.constructGeneralOperator(N, R, env)
  end
  # @show cond(convert(Matrix{Float64}, BigMatrix))
  return BigMatrix
end

"""Docstring for the function"""
function solve(N::Int64, R::Float64, env::Utils.SolutionEnvironment)::Vector{BigFloat}
  BigMatrix = constructOperatorFromEnv(N, R, env)
  BigRHS = zeros(N)
  BigRHS[1] = 1.0

  BigSolution = BigMatrix \ BigRHS
  return BigSolution / Utils.totalMass(BigSolution, env)
end

"""Docstring for the function"""
function solveWithRegularisation(N::Int64, R::Float64, s=1e-8, env::Utils.SolutionEnvironment)::Vector{BigFloat}
  A = constructOperatorFromEnv(N, R, env)
  BigMatrix = A' * A + s * LinearAlgebra.I
  BigRHS = zeros(N)
  BigRHS[1] = 1.0
  BigRHS = A' * BigRHS

  BigSolution = BigMatrix \ BigRHS
  return BigSolution / Utils.totalMass(BigSolution, env)
end

"""Docstring for the function"""
function outerOptimisation(N::Int64, env::Utils.SolutionEnvironment)
  F(R) = Utils.totalEnergy(solve(N, R, env), R, 0.0, env)
  f(x) = F(x[1])  # because optimize() only accepts vector inputs
  solution = Optim.optimize(f, [R0])
  return solution
end
end
