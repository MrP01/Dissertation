module GeneralKernelSolver
import ..Utils
import ..Solver
import ..Parameters, ..potentialFunction

"""Docstring for the function"""
function basisConversionMatrix(env::Utils.SolutionEnvironment)
  r = axes(env.P, 1)
  return mapreduce(permutedims, hcat, [env.P[:, 1:env.p.M] \ r .^ k for k in 0:env.p.M-1]')
end

"""Docstring for the function"""
function expandKernelInMonomials(env::Utils.SolutionEnvironment)
  r = axes(env.P, 1)
  BasisConversionMat = basisConversionMatrix(env)
  InteractionCoeffs = convert(Vector{Float64}, env.P[:, 1:env.p.M] \ potentialFunction.(r; pot=env.p.potential))  # in Jacobi basis
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
    operator += coefficient * R^power * Solver.constructOperator(N, float(power) + 0.001, env)
  end
  return operator
end

"""Docstring for the function"""
function solve(N::Int64, R::Float64, env::Utils.SolutionEnvironment)::Vector{BigFloat}
  BigMatrix = constructGeneralOperator(N, R, env)
  # @show cond(convert(Matrix{Float64}, BigMatrix))
  BigRHS = zeros(N)
  BigRHS[1] = 1.0

  BigSolution = BigMatrix \ BigRHS
  return BigSolution / Utils.totalMass(BigSolution, env)
end
end
