module AnalyticSolutions
import SpecialFunctions
import ..Params

function explicitSolution(x::AbstractVector{Float64}; p::Params.Parameters)
  if isa(p.potential, Params.AttractiveRepulsive)
    alpha, beta = p.potential.alpha, p.potential.beta
    if isapprox(alpha, 2.0; atol=1e-3) && -1.0 < beta < 2.0
      # Solution from Carillo, 2017-explicit-solutions
      prefactor = cos((2 - beta) * pi / 2) / ((beta - 1) * pi)
      R = (prefactor * SpecialFunctions.beta(0.5, (3 - beta) / 2))^(1 / (beta - 2))
      x *= R  # so this function accepts input from -1 to 1
      yOffset = prefactor * R^(1 - beta)  # simply the function evaluated at 0
      return R, prefactor * (R^2 .- x .^ 2) .^ ((1 - beta) / 2) * R^p.d
    end
  else
    error("I don't know of any solution with these parameters.")
  end
end
end
