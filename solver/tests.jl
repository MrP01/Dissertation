using Test

include("./utils.jl")
include("./solver.jl")
include("./generalkernel.jl")
import .Utils: recurrence, theorem216
import .Solver

@testset "Solver" begin
  @testset "recurrence relation" begin
    for n in 1:8
      r = rand()
      @test recurrence(theorem216(r, n - 1), theorem216(r, n), r, n) ≈ theorem216(r, n + 1) atol = 1e-16
    end
  end
  @testset "compare operator construction methods" begin
    for n in 1:5
      @test Solver.constructOperator(n, p.beta) ≈ Solver.recursivelyConstructOperatorWithReprojection(n, p.beta) atol = 1e-15
    end
  end
  @testset "solution is normalised" begin
    @test Utils.totalMass(Solver.solve(12)) ≈ 1.0 atol = 1e-16
  end
  @testset "jacobi to monomial basis conversion" begin
    monomialCoeffs = zeros(p.M)
    monomialCoeffs[1] = 0.3  # r^0 coefficient
    monomialCoeffs[3] = 2.0  # r^2 coefficient
    @test BasisConversionMat * monomialCoeffs ≈ Solver.defaultEnv.P[:, 1:p.M] \ (monomialCoeffs[1] * r .^ 0 + monomialCoeffs[3] * r .^ 2)

    jacobiCoeff = zeros(p.M)
    jacobiCoeff[1] = 3.0  # P_0^{(a, b)} coeff
    jacobiCoeff[3] = -1.0  # P_2^{(a, b)} coeff

    r_vec = 0:0.002:1
    @test sum((BasisConversionMat \ jacobiCoeff) .* [r_vec .^ k for k in 0:p.M-1], dims=1)[1] ≈
          vec(sum(jacobiCoeff .* Solver.defaultEnv.P[r_vec, 1:p.M]', dims=1))
  end
end

println("Keep testing!")
