using Test
include("./solver.jl")

@testset "Solver" begin
  @testset "recurrence relation" begin
    for n in 1:8
      r = rand()
      @test recurrence(theorem216(r, n - 1), theorem216(r, n), r, n) ≈ theorem216(r, n + 1)
    end
  end
  @testset "compare operator construction methods" begin
    for n in 1:5
      @test constructOperator(n, beta) ≈ recursivelyConstructOperatorWithReprojection(n, beta)
    end
  end
  @testset "solution is normalised" begin
    @test totalMass(solve(12)) ≈ 1.0 atol = 1e-15
  end
  @testset "jacobi to monomial basis conversion" begin
    monomialCoeffs = zeros(M)
    monomialCoeffs[1] = 0.3  # r^0 coefficient
    monomialCoeffs[3] = 2.0  # r^2 coefficient
    @test BasisConversionMat * monomialCoeffs ≈ P[:, 1:M] \ (monomialCoeffs[1] * r .^ 0 + monomialCoeffs[3] * r .^ 2)

    jacobiCoeff = zeros(M)
    jacobiCoeff[1] = 3.0  # P_0^{(a, b)} coeff
    jacobiCoeff[3] = -1.0  # P_2^{(a, b)} coeff
    @test sum((BasisConversionMat \ jacobiCoeff) .* [r_vec .^ k for k in 0:M-1], dims=1)[1] ≈
          vec(sum(jacobiCoeff .* P[r_vec, 1:M]', dims=1))
  end
end

println("Keep testing!")
