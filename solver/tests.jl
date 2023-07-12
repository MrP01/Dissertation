using Test

@test true
@testset "recurrence relation" begin
  for n in 1:8
    r = rand()
    @test recurrence(theorem216(r, n - 1), theorem216(r, n), r, n) ≈ theorem216(r, n + 1)
  end
end
