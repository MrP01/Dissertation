using CairoMakie
using SparseArrays
include("./solver.jl");

const RESULTS_FOLDER = joinpath(@__DIR__, "..", "figures", "results")

r_vec = 0:0.002:1
x_vec = -1:0.002:1

function plotDifferentOrderSolutions()
  fig = Figure()
  ax = Axis(fig[1, 1])
  x_vec_noends = x_vec[2:end-1]
  # lines!(ax, x_vec_noends, obtainMeasure(x_vec_noends, 2), label="N = 2")
  lines!(ax, x_vec_noends, rho(x_vec_noends, solve(3)), label="N = 3")
  lines!(ax, x_vec_noends, rho(x_vec_noends, solve(4)), label="N = 4")
  lines!(ax, x_vec_noends, rho(x_vec_noends, solve(5)), label="N = 5")
  lines!(ax, x_vec_noends, rho(x_vec_noends, solve(6)), label="N = 6")
  lines!(ax, x_vec_noends, rho(x_vec_noends, solve(7)), label="N = 7")
  axislegend(ax)
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
