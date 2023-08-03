module AnalyticSolutions
import SpecialFunctions
import ..Parameters, ..defaultParams

function explicitSolution(x::Real, p::Parameters=defaultParams)
  if isapprox(p.alpha, 2.0; atol=1e-3) && -1.0 < p.beta < 2.0
    # Solution from Carillo, 2017-explicit-solutions
    prefactor = cos((2 - p.beta) * pi / 2) / ((p.beta - 1) * pi)
    R = (prefactor * SpecialFunctions.beta(0.5, (3 - p.beta) / 2))^(1 / (p.beta - 2))
    x *= R  # so this function accepts input from -1 to 1
    yOffset = prefactor * R^(1 - p.beta)  # simply the function evaluated at 0
    return prefactor * (R^2 - x^2)^((1 - p.beta) / 2) - yOffset
  else
    error("I don't know of any solution with these parameters.")
  end
end
end
