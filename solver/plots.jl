using GLMakie
using SparseArrays
include("./solver.jl");

const RESULTS_FOLDER = joinpath(@__DIR__, "..", "figures", "results")

r_vec = 0:0.01:1
x_vec = -1:0.01:1

function plotDifferentOrderSolutions()
  # y_vec_radial = vec(sum(BigSolution .* P[r_vec, 1:N]', dims=1));
  p = lines(x_vec, vec(sum(solve(2) .* P[abs.(x_vec), 1:2]', dims=1)), label="N = 2")
  lines!(x_vec, vec(sum(solve(3) .* P[abs.(x_vec), 1:3]', dims=1)), label="N = 3")
  lines!(x_vec, vec(sum(solve(4) .* P[abs.(x_vec), 1:4]', dims=1)), label="N = 4")
  lines!(x_vec, vec(sum(solve(5) .* P[abs.(x_vec), 1:5]', dims=1)), label="N = 5")
  save(joinpath(RESULTS_FOLDER, "solution-increasing-order.png"))
  return p
end

function plotOperators()
  fig = Figure(resolution=(890, 460))
  ax = Axis(fig[1, 1], yreversed=true)
  spy!(ax, sparse(log10.(abs.(constructOperator(20, alpha)))), markersize=20, markershape=:rect)
  ax = Axis(fig[1, 2], yreversed=true)
  spy!(ax, sparse(log10.(abs.(constructOperator(20, beta)))), markersize=20, markershape=:rect)
  save(joinpath(RESULTS_FOLDER, "attractive-repulsive-operator.png"), fig)
  return fig
end

function plotSpatialEnergyDependence()
  solution = solve(5)
  TE(r) = totalEnergy(solution, r)
  p = lines(r_vec[1:end-1], TE.(r_vec[1:end-1]))
  save(joinpath(RESULTS_FOLDER, "energy-dependence-on-r.png"))
  return p
end

println("All plotted for today?")
