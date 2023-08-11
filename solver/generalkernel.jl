module GeneralKernelSolver
import ..Params
import ..Utils
import ..AttractiveRepulsiveSolver

"""Docstring for the function. M: number of basis elements to expand the general kernel in."""
function basisConversionMatrix(env::Utils.SolutionEnvironment, M=5)
  r = axes(env.P, 1)
  return mapreduce(permutedims, hcat, [env.P[:, 1:M] \ r .^ k for k in 0:M-1]')
end

"""Docstring for the function. M: number of basis elements to expand the general kernel in."""
function expandKernelInMonomials(env::Utils.SolutionEnvironment, M=5)
  r = axes(env.P, 1)
  BasisConversionMat = basisConversionMatrix(env)
  InteractionCoeffs = convert(Vector{Float64}, env.P[:, 1:M] \ Params.potentialFunction.(r; pot=env.p.potential))  # in Jacobi basis
  MonomialInteractionCoeffs = BasisConversionMat \ InteractionCoeffs  # in monomial basis
  return MonomialInteractionCoeffs
end

"""Docstring for the function"""
function constructGeneralOperator(N::Int64, R::Float64, env::Utils.SolutionEnvironment)
  monomial = expandKernelInMonomials(env)
  operator = zeros(N, N)
  for index in eachindex(monomial)
    # index starts from 1
    power, coefficient = index - 1, monomial[index]
    operator += coefficient * R^power * AttractiveRepulsiveSolver.constructOperator(N, float(power) + 0.001, env)
  end
  return operator
end
end
