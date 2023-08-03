module GeneralKernelSolver
import ..Utils
import ..Parameters, ..defaultParams

B, P = Utils.createBasis()
r = axes(P, 1)

# expand the interaction potential in the basis
BasisConversionMat = mapreduce(permutedims, hcat, [P[:, 1:defaultParams.M] \ r .^ k for k in 0:defaultParams.M-1]')
InteractionCoeffs = convert(Vector{Float64}, P[:, 1:defaultParams.M] \ defaultParams.InteractionPotential.(r))  # in Jacobi basis
MonomialInteractionCoeffs = BasisConversionMat \ InteractionCoeffs  # in monomial basis
end
