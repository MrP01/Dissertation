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
      @test constructOperator(n, beta) ≈ recursivelyConstructOperator(n, beta)
    end
  end
end

println("Keep testing!")
