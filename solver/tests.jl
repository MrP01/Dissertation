using Test

include("./parameters.jl")
include("./solver.jl")

import .Params
import .Solver
import .AttractiveRepulsiveSolver

@testset "Solver" begin
  @testset "recurrence relation" begin
    p = Params.defaultParams
    for n in 1:8
      r = rand()
      beta = p.potential.beta
      @test AttractiveRepulsiveSolver.recurrence(
        r;
        oldestValue=Utils.theorem216(r; n=n - 1, beta=beta, p=p),
        oldValue=Utils.theorem216(r; n=n, beta=beta, p=p),
        n=n, beta=beta, p=p
      ) ≈ Utils.theorem216(r; n=n + 1, beta=beta, p=p) atol = 1e-16
    end
  end
  @testset "compare operator construction methods" begin
    env = Utils.defaultEnv
    for n in 1:2
      @test AttractiveRepulsiveSolver.constructOperator(n, env.p.potential.beta, env) ≈
            AttractiveRepulsiveSolver.recursivelyConstructOperatorWithReprojection(n, env.p.potential.beta, env) atol = 1e-15
    end
  end
  @testset "solution is normalised" begin
    env = Utils.defaultEnv
    @test Utils.totalMass(Solver.solve(12, env.p.R0, env), env) ≈ 1.0 atol = 1e-16
  end
  @testset "jacobi to monomial basis conversion" begin
    M = 5
    env = Utils.defaultEnv
    monomialCoeffs = zeros(M)
    monomialCoeffs[1] = 0.3  # r^0 coefficient
    monomialCoeffs[3] = 2.0  # r^2 coefficient
    r = axes(env.P, 1)
    @test Utils.basisConversionMatrix(env.P, M) * monomialCoeffs ≈
          env.P[:, 1:M] \ (monomialCoeffs[1] * r .^ 0 + monomialCoeffs[3] * r .^ 2)

    jacobiCoeff = zeros(M)
    jacobiCoeff[1] = 3.0  # P_0^{(a, b)} coeff
    jacobiCoeff[3] = -1.0  # P_2^{(a, b)} coeff

    r_vec = 0:0.002:1
    @test sum((Utils.basisConversionMatrix(env.P, M) \ jacobiCoeff) .* [r_vec .^ k for k in 0:M-1], dims=1)[1] ≈
          vec(sum(jacobiCoeff .* env.P[r_vec, 1:M]', dims=1))
  end
end

println("Keep testing!")
