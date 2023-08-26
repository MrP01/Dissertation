using Test

include("./parameters.jl")
include("./solver.jl")
include("./analyticsolutions.jl")

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
    for n in 2 .^ (1:6)
      @test AttractiveRepulsiveSolver.constructOperator(n, env.p.potential.beta, env) ≈
            AttractiveRepulsiveSolver.recursivelyConstructOperator(n, env.p.potential.beta, env) atol = 1e-12
    end
  end
  @testset "solution is normalised" begin
    env = Utils.defaultEnv
    @test Utils.totalMass(Solver.solve(12, env.p.R0, env), env) ≈ 1.0 atol = 1e-16
  end
  @testset "jacobi to monomial basis conversion" begin
    G = 5
    env = Utils.defaultEnv
    monomialCoeffs = zeros(G)
    monomialCoeffs[1] = 0.3  # r^0 coefficient
    monomialCoeffs[3] = 2.0  # r^2 coefficient
    r = axes(env.P, 1)
    @test Utils.basisConversionMatrix(env.P, G) * monomialCoeffs ≈
          env.P[:, 1:G] \ (monomialCoeffs[1] * r .^ 0 + monomialCoeffs[3] * r .^ 2)

    jacobiCoeff = zeros(G)
    jacobiCoeff[1] = 3.0  # P_0^{(a, b)} coeff
    jacobiCoeff[3] = -1.0  # P_2^{(a, b)} coeff

    r_vec = 0:0.002:1
    @test sum((Utils.basisConversionMatrix(env.P, G) \ jacobiCoeff) .* [r_vec .^ k for k in 0:G-1], dims=1)[1] ≈
          vec(sum(jacobiCoeff .* env.P[r_vec, 1:G]', dims=1))
  end
  @testset "N=1 approximation to R_opt works" begin
    N = 1
    env = Utils.defaultEnv
    R_opt = Solver.outerOptimisation(N, env).minimizer[1]
    @test R_opt ≈ Solver.guessSupportRadius(N; p=env.p) atol = 1e-6
  end
  @testset "analytic solution is close" begin
    p = Params.knownAnalyticParams
    env = Utils.createEnvironment(p)
    r_vec_noend = r_vec[1:end-1]
    R, analytic = AnalyticSolutions.explicitSolution(r_vec_noend; p=p)
    @test sum(abs.(Utils.rho(r_vec_noend, Solver.solve(30, R, env), env) .- analytic)) / length(r_vec_noend) < 1e-3
  end
end

println("Keep testing!")
