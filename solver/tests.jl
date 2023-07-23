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
end

println("Keep testing!")
