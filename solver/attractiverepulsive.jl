module AttractiveRepulsiveSolver
import ContinuumArrays: MappedWeightedBasisLayout, WeightedBasisLayout
import ClassicalOrthogonalPolynomials: jacobimatrix
import LinearAlgebra: cond, I
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

function recurrence(r; oldestValue, oldValue, n, beta, p::Params.Parameters)
  # using Corollary 2.18
  m = p.m
  @assert isa(p.potential, Params.AttractiveRepulsive)
  alpha = p.potential.alpha
  c_a = -((-alpha + 2m + 4n) * (-alpha + 2m + 4n + 2) * (alpha + p.d - 2 * (p.m + n + 1))) /
        (2 * (n + 1) * (-alpha + beta + 2m + 2n + 2) * (-alpha + beta + p.d + 2m + 2n))
  c_b = -((-alpha + 2m + 4n) * (alpha + p.d - 2(p.m + n + 1)) * (p.d * (-alpha + 2 * beta + 2m + 2) - 2 * (2n - beta) * (-alpha + beta + 2m + 2n))) /
        (2 * (n + 1) * (-alpha + 2m + 4n - 2) * (-alpha + beta + 2m + 2n + 2) * (-alpha + beta + p.d + 2m + 2n))
  c_c = ((-beta + 2n - 2) * (beta + p.d - 2n) * (-alpha + 2m + 4n + 2) * (alpha + p.d - 2 * (p.m + n)) * (alpha + p.d - 2 * (p.m + n + 1))) /
        (4n * (n + 1) * (-alpha + 2m + 4n - 2) * (-alpha + beta + 2m + 2n + 2) * (-alpha + beta + p.d + 2m + 2n))
  return (c_a * r^2 + c_b) * oldValue + c_c * oldestValue
end

# from https://github.com/JuliaApproximation/EquilibriumMeasures.jl/blob/master/examples/powerlaw_examples.jl
function coeffRecurrenceCoeffs(α, β, d, m, n, z)
  return (
    -(((2m + 4n - α) * (d - 2 * (1 + m + n) + α) *
       (-8 * n^2 * I - 4 * z + 4m^2 * z + 16 * n^2 * z + 4n * α * I - 8 * n * z * α + z * α^2 - 2 * α * β * I + 2 * β^2 * I +
        4m * (-2n * I + 4n * z - z * α + β * I) + d * I * (2 + 2m - α + 2 * β))) /
      (2 * (1 + n) * (-2 + 2m + 4n - α) * (2 + 2m + 2n - α + β) * (d + 2m + 2n - α + β))),
    (((2 + 2m + 4n - α) * (d - 2 * (m + n) + α) * (d - 2 * (1 + m + n) + α) * (-2 + 2n - β) * (d - 2n + β)) /
     (4n * (1 + n) * (-2 + 2m + 4n - α) * (2 + 2m + 2n - α + β) * (d + 2m + 2n - α + β)))
  )
end

RecOpMem = LRUCache.LRU{Tuple{Int64,Float64,SolutionEnvironment},Matrix{BigFloat}}(maxsize=20)
function recursivelyConstructOperator(N::Int64, beta::Float64, env::SolutionEnvironment)::Matrix{BigFloat}
  p = env.p
  get!(RecOpMem, (N, beta, env)) do
    if isa(p.potential, Params.AttractiveRepulsive)
      alpha = p.potential.alpha
    elseif isa(p.potential, Params.MorsePotential)
      alpha = Utils.GENERAL_ALPHA
    else
      error("Unkown potential")
    end
    r_axis = axes(env.P, 1)
    Mat = zeros(BigFloat, N, N)
    OldestCoeff = env.P[:, 1:N] \ Utils.theorem216.(r_axis; n=0, beta=beta, p=env.p)
    OldCoeff = env.P[:, 1:N] \ Utils.theorem216.(r_axis; n=1, beta=beta, p=env.p)

    Mat[:, 1] = OldestCoeff
    if N < 2
      return Mat
    end

    Mat[:, 2] = OldCoeff
    if N < 3
      return Mat
    end

    Z = ((jacobimatrix(env.B)+I))[1:N, 1:N] ./ 2  # == (X+1)/2, the argument!
    for remainingColumn in 3:N
      n = remainingColumn - 2
      ca, cc = coeffRecurrenceCoeffs(alpha, beta, env.p.d, env.p.m, n, Z)
      Coeff = ca * OldCoeff + cc * OldestCoeff
      Mat[:, remainingColumn] = Coeff
      OldestCoeff = OldCoeff
      OldCoeff = Coeff
    end
    Utils.zeroOutTinyValues!(Mat)
    Mat
  end
end

function constructFullOperator(N::Int64, R::Float64, env::SolutionEnvironment)::Matrix{BigFloat}
  p::Params.Parameters = env.p
  @assert isa(p.potential, Params.AttractiveRepulsive)
  alpha, beta, d = p.potential.alpha, p.potential.beta, p.d
  AttractiveMatrix = recursivelyConstructOperator(N, alpha, env)
  RepulsiveMatrix = recursivelyConstructOperator(N, beta, env)
  # @show cond(convert(Matrix{Float64}, AttractiveMatrix))
  # @show cond(convert(Matrix{Float64}, RepulsiveMatrix))
  d = 0
  return (R^(alpha + d) / alpha) * AttractiveMatrix - (R^(beta + d) / beta) * RepulsiveMatrix
end
end
