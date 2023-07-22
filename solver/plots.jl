using CairoMakie
using SparseArrays
include("./solver.jl");

const RESULTS_FOLDER = joinpath(@__DIR__, "..", "figures", "results")

r_vec = 0:0.01:1
x_vec = -1:0.01:1

function plotDifferentOrderSolutions()
  # y_vec_radial = vec(sum(BigSolution .* P[r_vec, 1:N]', dims=1));
  fig = Figure()
  lines(fig[1, 1], x_vec, vec(sum(solve(2) .* P[abs.(x_vec), 1:2]', dims=1)), label="N = 2")
  lines!(fig[1, 1], x_vec, vec(sum(solve(3) .* P[abs.(x_vec), 1:3]', dims=1)), label="N = 3")
  lines!(fig[1, 1], x_vec, vec(sum(solve(4) .* P[abs.(x_vec), 1:4]', dims=1)), label="N = 4")
  lines!(fig[1, 1], x_vec, vec(sum(solve(5) .* P[abs.(x_vec), 1:5]', dims=1)), label="N = 5")
  save(joinpath(RESULTS_FOLDER, "solution-increasing-order.pdf"), fig)
  return fig
end

function plotOperators()
  op1 = constructOperator(20, alpha)
  op2 = constructOperator(20, beta)
  fig = Figure(resolution=(920, 460))
  ax = Axis(fig[1, 1], yreversed=true, title=L"\text{Attractive Operator}~(\alpha = %$alpha)")
  spy!(ax, sparse(log10.(abs.(op1))), marker=:rect, markersize=32, framesize=0)
  ax = Axis(fig[1, 2], yreversed=true, title=L"\text{Repulsive Operator}~(\beta = %$beta)")
  spy!(ax, sparse(log10.(abs.(op2))), marker=:rect, markersize=32, framesize=0)
  save(joinpath(RESULTS_FOLDER, "attractive-repulsive-operator.pdf"), fig)
  return fig
end

function plotSpatialEnergyDependence()
  solution = solve(5)
  fig = Figure()
  TE(r) = totalEnergy(solution, r)
  lines(fig[1, 1], r_vec[1:end-1], TE.(r_vec[1:end-1]))
  save(joinpath(RESULTS_FOLDER, "energy-dependence-on-r.pdf"), fig)
  return fig
end

println("All plotted for today?")
