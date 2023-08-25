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
function solve(N::Int64, R::Float64, env::Utils.SolutionEnvironment)::Vector{BigFloat}
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

"""Docstring for the function"""
function outerOptimisation(N::Int64, env::Utils.SolutionEnvironment, method=Optim.NewtonTrustRegion(), R0=0)
  if R0 == 0
    R0 = env.p.R0
  end
  F(R) = Utils.totalEnergy(solve(N, R, env), R, 0.0, env)
  f(x) = F(x[1])  # because optimize() only accepts vector inputs
  solution = Optim.optimize(f, [R0], method=method)
  return solution
end

function guessSupportRadius(N; p::Params.Parameters)
  # This works for N=1. For N >= 2, not so much.
  alpha, beta = p.potential.alpha, p.potential.beta
  f_a, f_b = 0.0, 0.0
  for n in 0:N-1
    f_a += Float64.(Utils.theorem216.(0.0; n=n, beta=alpha, p=p))
    f_b += Float64.(Utils.theorem216.(0.0; n=n, beta=beta, p=p))
  end
  R = (f_b / f_a)^(1 / (alpha - beta))
  return R
end

function guessSupportRadiusEvenHarder(solution::Vector{BigFloat}; p::Params.Parameters)
  # no, does not work
  alpha, beta = p.potential.alpha, p.potential.beta
  U_a, U_b = 0.0, 0.0
  for k in eachindex(solution)
    U_a += solution[k] * Float64.(Utils.theorem216.(0.0; n=k - 1, beta=alpha, p=p))
    U_b += solution[k] * Float64.(Utils.theorem216.(0.0; n=k - 1, beta=beta, p=p))
  end
  R = (U_b / U_a)^(1 / (alpha - beta))
  return R
end
end
