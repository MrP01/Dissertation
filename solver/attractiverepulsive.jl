module AttractiveRepulsiveSolver
import ContinuumArrays: MappedWeightedBasisLayout, WeightedBasisLayout
import LinearAlgebra: cond
import SpecialFunctions: gamma
import LRUCache
import Optim

import ..Params
import ..Utils
import ..Utils: SolutionEnvironment

OpMem = LRUCache.LRU{Tuple{Int64,Float64,SolutionEnvironment},Matrix{BigFloat}}(maxsize=20)
"""Creates an operator given alpha and beta. Caches it."""
function constructOperator(N::Int64, beta::Float64, env::SolutionEnvironment)::Matrix{BigFloat}
  get!(OpMem, (N, beta, env)) do
    Mat = zeros(BigFloat, N, N)
    r_axis = axes(env.P, 1)
    for n in 0:N-1
      Function = Utils.theorem216.(r_axis; n=n, beta=beta, p=env.p)
      ExpansionCoeffs = env.P[:, 1:N] \ Function  # expands the function in the P basis
      Mat[:, n+1] .= ExpansionCoeffs
    end
    Utils.zeroOutTinyValues!(Mat)
    Mat
  end
end

"""Recursively constructs with reprojection, terrible because the types keep on nesting inside of one another."""
function recursivelyConstructOperatorWithReprojection(N::Int64, beta::Float64, env::SolutionEnvironment)::Matrix{BigFloat}
  Mat = zeros(BigFloat, N, N)
  r_axis = axes(env.P, 1)
  OldestFunction = Utils.theorem216.(r_axis; n=0, beta=beta, p=env.p)
  OldFunction = Utils.theorem216.(r_axis; n=1, beta=beta, p=env.p)

  Mat[:, 1] = env.P[:, 1:N] \ OldestFunction
  if N < 2
    return Mat
  end

  Mat[:, 2] = env.P[:, 1:N] \ OldFunction
  if N < 3
    return Mat
  end

  for remainingColumn in 3:N
    Function = Utils.recurrence.(r_axis; oldestValue=OldestFunction, oldValue=OldFunction, n=remainingColumn - 2, beta=beta, p=env.p)
    Mat[:, remainingColumn] = env.P[:, 1:N] \ Function
    OldestFunction = OldFunction
    OldFunction = Function
  end
  Utils.zeroOutTinyValues!(Mat)
  return Mat
end

function constructFullOperator(N::Int64, R::Float64, env::SolutionEnvironment)::Matrix{BigFloat}
  p::Params.Parameters = env.p
  @assert isa(p.potential, Params.AttractiveRepulsive)
  AttractiveMatrix = constructOperator(N, p.potential.alpha, env)
  RepulsiveMatrix = constructOperator(N, p.potential.beta, env)
  # @show cond(convert(Matrix{Float64}, AttractiveMatrix))
  # @show cond(convert(Matrix{Float64}, RepulsiveMatrix))
  return (R^p.potential.alpha / p.potential.alpha) * AttractiveMatrix - (R^p.potential.beta / p.potential.beta) * RepulsiveMatrix
end
end
