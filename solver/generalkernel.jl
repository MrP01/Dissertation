module GeneralKernelSolver
import ..Params
import ..Utils
import ..AttractiveRepulsiveSolver

"""Docstring for the function"""
function constructGeneralOperator(N::Int64, R::Float64, env::Utils.SolutionEnvironment)
  operator = zeros(N, N)
  for index in eachindex(env.monomial)
    # index starts from 1!
    power, coefficient = index - 1, env.monomial[index]
    operator += coefficient * R^power * AttractiveRepulsiveSolver.constructOperator(N, float(power), env)
    # TODO: should we divide by power? i.e. R^power / power
  end
  return operator
end
end
