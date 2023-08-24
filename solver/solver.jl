include("./utils.jl")
include("./attractiverepulsive.jl")
include("./generalkernel.jl")

module Solver
import LinearAlgebra
import LRUCache
import Optim
import ..Params
import ..Utils
import ..AttractiveRepulsiveSolver
import ..GeneralKernelSolver

FullOpMem = LRUCache.LRU{Tuple{Int64,Float64,Utils.SolutionEnvironment},Matrix{BigFloat}}(maxsize=20)
function constructOperatorFromEnv(N::Int64, R::Float64, env::Utils.SolutionEnvironment)
  get!(FullOpMem, (N, R, env)) do
    if isa(env.p.potential, Params.AttractiveRepulsive)
      BigMatrix = AttractiveRepulsiveSolver.constructFullOperator(N, R, env)
    else
      BigMatrix = GeneralKernelSolver.constructGeneralOperator(N, R, env)
    end
    # preConditioner = 1 / (2^(N - 1) * factorial(N - 1)) * LinearAlgebra.diagm(0 => 1 ./ (N .+ (0:N-1)))
    # preConditioner = LinearAlgebra.diagm(0 => 1 ./ (N .+ (0:N-1)))
    Utils.zeroOutTinyValues!(BigMatrix)
    BigMatrix
  end
end

"""Docstring for the function"""
function solveWithoutRegularisation(N::Int64, R::Float64, env::Utils.SolutionEnvironment)::Vector{BigFloat}
  BigMatrix = constructOperatorFromEnv(N, R, env)
  BigRHS = zeros(N)
  BigRHS[1] = 1.0

  BigSolution = BigMatrix \ BigRHS
  return BigSolution / Utils.totalMass(BigSolution, env)
end

"""Docstring for the function"""
function solveWithRegularisation(N::Int64, R::Float64, env::Utils.SolutionEnvironment, s::Float64)::Vector{BigFloat}
  A = constructOperatorFromEnv(N, R, env)
  BigMatrix = A' * A + s * LinearAlgebra.I
  BigRHS = zeros(N)
  BigRHS[1] = 1.0
  BigRHS = A' * BigRHS

  BigSolution = BigMatrix \ BigRHS
  return BigSolution / Utils.totalMass(BigSolution, env)
end

function solve(N::Int64, R::Float64, env::Utils.SolutionEnvironment)::Vector{BigFloat}
  return solveWithRegularisation(N, R, env, env.p.s0)
end

"""Docstring for the function"""
function outerOptimisation(N::Int64, env::Utils.SolutionEnvironment, method=Optim.LBFGS())
  F(R) = Utils.totalEnergy(solve(N, R, env), R, 0.0, env)
  f(x) = F(x[1])  # because optimize() only accepts vector inputs
  solution = Optim.optimize(f, [env.p.R0], method=method)
  return solution
end
end
