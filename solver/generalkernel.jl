include("./parameters.jl")
import .Solver

B, P = Solver.createBasis()
r = axes(P, 1)
# expand the interaction potential in the basis
BasisConversionMat = mapreduce(permutedims, hcat, [P[:, 1:p.M] \ r .^ k for k in 0:p.M-1]')
InteractionCoeffs = convert(Vector{Float64}, P[:, 1:p.M] \ p.InteractionPotential.(r))  # in Jacobi basis
MonomialInteractionCoeffs = BasisConversionMat \ InteractionCoeffs  # in monomial basis
