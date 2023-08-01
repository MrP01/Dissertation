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
  op1 = constructOperator(30, alpha)
  op2 = constructOperator(30, beta)
  fig = Figure(resolution=(920, 400))
  ax = Axis(fig[1, 1][1, 1], yreversed=true, title=L"\text{Attractive Operator}~(\alpha = %$alpha)")
  s = spy!(ax, sparse(log10.(abs.(op1))), marker=:rect, markersize=32, framesize=0)
  Colorbar(fig[1, 1][1, 2], s, flipaxis=false)
  ax = Axis(fig[1, 2][1, 1], yreversed=true, title=L"\text{Repulsive Operator}~(\beta = %$beta)")
  s = spy!(ax, sparse(log10.(abs.(op2))), marker=:rect, markersize=32, framesize=0)
  Colorbar(fig[1, 2][1, 2], s, flipaxis=false)
  save(joinpath(RESULTS_FOLDER, "attractive-repulsive-operator.pdf"), fig)
  return fig
end

function plotSpatialEnergyDependence()
  fig = Figure()
  ax = Axis(fig[1, 1])
  for N in 4:4:20
    solution = solve(N)
    TE(r) = totalEnergy(solution, r)
    lines!(ax, r_vec[1:end-1], TE.(r_vec[1:end-1]), label=L"N = %$N")
  end
  axislegend(ax)
  save(joinpath(RESULTS_FOLDER, "energy-dependence-on-r.pdf"), fig)
  return fig
end

function plotSolutionConvergence()
  Ns = 3:1:22
  errors = zeros(length(Ns))
  previous = rho(r_vec[1:end-1], solve(2))
  for k in eachindex(Ns)
    N = Ns[k]
    solution = solve(N)
    this = rho(r_vec[1:end-1], solution)
    errors[k] = sum((this - previous) .^ 2)
    previous = this
  end

  fig = Figure()
  ax = Axis(fig[1, 1], yscale=log10, xlabel=L"N", ylabel="Squared Error")
  lines!(ax, Ns, errors)
  scatter!(ax, Ns, errors, color=:red)
  save(joinpath(RESULTS_FOLDER, "convergence.pdf"), fig)
  return fig
end

function plotAll()
  plotDifferentOrderSolutions()
  plotOperators()
  plotSolutionConvergence()
  plotSpatialEnergyDependence()
  return
end

println("All plotted for today?")
