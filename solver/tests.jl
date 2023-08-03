using Test

include("./parameters.jl")
include("./utils.jl")
include("./solver.jl")
include("./generalkernel.jl")
import .Solver

@testset "Solver" begin
  @testset "recurrence relation" begin
    for n in 1:8
      r = rand()
      @test Utils.recurrence(Utils.theorem216(r, n - 1), Utils.theorem216(r, n), r, n) ≈ Utils.theorem216(r, n + 1) atol = 1e-16
    end
  end
  @testset "compare operator construction methods" begin
    for n in 1:5
      @test Solver.constructOperator(n, defaultParams.beta) ≈
            Solver.recursivelyConstructOperatorWithReprojection(n, defaultParams.beta) atol = 1e-15
    end
  end
  @testset "solution is normalised" begin
    @test Utils.totalMass(Solver.solve(12)) ≈ 1.0 atol = 1e-16
  end
  @testset "jacobi to monomial basis conversion" begin
    monomialCoeffs = zeros(defaultParams.M)
    monomialCoeffs[1] = 0.3  # r^0 coefficient
    monomialCoeffs[3] = 2.0  # r^2 coefficient
    r = axes(GeneralKernelSolver.P, 1)
    @test GeneralKernelSolver.BasisConversionMat * monomialCoeffs ≈
          Solver.defaultEnv.P[:, 1:defaultParams.M] \ (monomialCoeffs[1] * r .^ 0 + monomialCoeffs[3] * r .^ 2)

    jacobiCoeff = zeros(defaultParams.M)
    jacobiCoeff[1] = 3.0  # P_0^{(a, b)} coeff
    jacobiCoeff[3] = -1.0  # P_2^{(a, b)} coeff

    r_vec = 0:0.002:1
    @test sum((GeneralKernelSolver.BasisConversionMat \ jacobiCoeff) .* [r_vec .^ k for k in 0:defaultParams.M-1], dims=1)[1] ≈
          vec(sum(jacobiCoeff .* Solver.defaultEnv.P[r_vec, 1:defaultParams.M]', dims=1))
  end
end

println("Keep testing!")
