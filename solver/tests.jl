using Test

include("./parameters.jl")
include("./utils.jl")
include("./solver.jl")
include("./generalkernel.jl")

@testset "Solver" begin
  # @testset "recurrence relation" begin
  #   for n in 1:8
  #     r = rand()
  #     beta = defaultParams.beta
  #     @test Utils.recurrence(
  #       r;
  #       oldestValue=Utils.theorem216(r; n=n - 1, beta=beta, p=defaultParams),
  #       oldValue=Utils.theorem216(r; n=n, beta=beta, p=defaultParams),
  #       n=n, beta=beta, p=defaultParams
  #     ) ≈ Utils.theorem216(r; n=n + 1, beta=beta, p=defaultParams) atol = 1e-16
  #   end
  # end
  # @testset "compare operator construction methods" begin
  #   for n in 1:2
  #     @test Solver.constructOperator(n, defaultParams.beta, Utils.defaultEnv) ≈
  #           Solver.recursivelyConstructOperatorWithReprojection(n, defaultParams.beta, Utils.defaultEnv) atol = 1e-15
  #   end
  # end
  # @testset "solution is normalised" begin
  #   @test Utils.totalMass(Solver.solve(12, defaultParams.R0, Utils.defaultEnv), Utils.defaultEnv) ≈ 1.0 atol = 1e-16
  # end
  @testset "jacobi to monomial basis conversion" begin
    env = Utils.createEnvironment(defaultParams)
    monomialCoeffs = zeros(env.p.M)
    monomialCoeffs[1] = 0.3  # r^0 coefficient
    monomialCoeffs[3] = 2.0  # r^2 coefficient
    r = axes(env.P, 1)
    @test GeneralKernelSolver.basisConversionMatrix(env) * monomialCoeffs ≈
          env.P[:, 1:env.p.M] \ (monomialCoeffs[1] * r .^ 0 + monomialCoeffs[3] * r .^ 2)

    jacobiCoeff = zeros(env.p.M)
    jacobiCoeff[1] = 3.0  # P_0^{(a, b)} coeff
    jacobiCoeff[3] = -1.0  # P_2^{(a, b)} coeff

    r_vec = 0:0.002:1
    @test sum((GeneralKernelSolver.basisConversionMatrix(env) \ jacobiCoeff) .* [r_vec .^ k for k in 0:env.p.M-1], dims=1)[1] ≈
          vec(sum(jacobiCoeff .* env.P[r_vec, 1:env.p.M]', dims=1))
  end
end

println("Keep testing!")
