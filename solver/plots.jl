using CairoMakie
using SparseArrays

include("./parameters.jl")
include("./utils.jl")
include("./solver.jl")
include("./analyticsolutions.jl")
import .Solver
import .AnalyticSolutions

const RESULTS_FOLDER = joinpath(@__DIR__, "..", "figures", "results")

r_vec = 0:0.002:1
r_vec_noend = r_vec[1:end-1]
x_vec = -1:0.002:1
x_vec_noends = x_vec[2:end-1]
pow10tickformat(values) = [L"10^{%$(Int(value))}" for value in values]

function plotDifferentOrderSolutions()
  fig = Figure()
  ax = Axis(fig[1, 1])
  R = defaultParams.R0
  env = Utils.defaultEnv
  # lines!(ax, x_vec_noends, obtainMeasure(x_vec_noends, 2), label="N = 2")
  lines!(ax, x_vec_noends, Utils.rho(x_vec_noends, Solver.solve(3, R, env), env), label="N = 3")
  lines!(ax, x_vec_noends, Utils.rho(x_vec_noends, Solver.solve(4, R, env), env), label="N = 4")
  lines!(ax, x_vec_noends, Utils.rho(x_vec_noends, Solver.solve(5, R, env), env), label="N = 5")
  lines!(ax, x_vec_noends, Utils.rho(x_vec_noends, Solver.solve(6, R, env), env), label="N = 6")
  lines!(ax, x_vec_noends, Utils.rho(x_vec_noends, Solver.solve(7, R, env), env), label="N = 7")
  axislegend(ax)
  save(joinpath(RESULTS_FOLDER, "solution-increasing-order.pdf"), fig)
  return fig
end

function plotOperators(N=30)
  alpha, beta = defaultParams.alpha, defaultParams.beta
  op1 = Solver.constructOperator(N, alpha, Utils.defaultEnv)
  op2 = Solver.constructOperator(N, beta, Utils.defaultEnv)
  fig = Figure(resolution=(920, 400))
  ax = Axis(fig[1, 1][1, 1], yreversed=true, title=L"\text{Attractive Operator}~(\alpha = %$alpha)")
  s = spy!(ax, sparse(log10.(abs.(op1))), marker=:rect, markersize=32, framesize=0)
  Colorbar(fig[1, 1][1, 2], s, flipaxis=false, tickformat=pow10tickformat)
  ax = Axis(fig[1, 2][1, 1], yreversed=true, title=L"\text{Repulsive Operator}~(\beta = %$beta)")
  s = spy!(ax, sparse(log10.(abs.(op2))), marker=:rect, markersize=32, framesize=0)
  Colorbar(fig[1, 2][1, 2], s, flipaxis=false, tickformat=pow10tickformat)
  save(joinpath(RESULTS_FOLDER, "attractive-repulsive-operator.pdf"), fig)
  return fig
end

function plotSpatialEnergyDependence()
  fig = Figure()
  ax = Axis(fig[1, 1])
  for R in 0.4:0.1:1.2
    solution = Solver.solve(12, R, Utils.defaultEnv)
    TE(r) = Utils.totalEnergy(solution, R, r, Utils.defaultEnv)
    lines!(ax, r_vec[1:end-1], TE.(r_vec[1:end-1]), label=L"R = %$R")
  end
  axislegend(ax)
  save(joinpath(RESULTS_FOLDER, "energy-dependence-on-r.pdf"), fig)
  return fig
end

function plotStepByStepConvergence()
  Ns = 1:1:22
  env = Utils.defaultEnv
  errors = zeros(length(Ns))
  R = Utils.defaultParams.R0
  best = Utils.rho(r_vec[1:end-1], Solver.solve(24, R, env), env)
  for k in eachindex(Ns)
    N = Ns[k]
    solution = Solver.solve(N, R, env)
    this = Utils.rho(r_vec[1:end-1], solution, env)
    errors[k] = sum((this - best) .^ 2) / length(r_vec)
  end

  fig = Figure()
  ax = Axis(fig[1, 1], yscale=log10, xlabel=L"N", ylabel="Squared Error")
  lines!(ax, Ns, errors)
  scatter!(ax, Ns, errors, color=:red)
  save(joinpath(RESULTS_FOLDER, "convergence.pdf"), fig)
  return fig
end

function plotOuterOptimisation()
  R_vec = 0.98:0.0001:1.15
  env = Utils.defaultEnv
  F(R) = Utils.totalEnergy(Solver.solve(8, R, env), R, 0.0, env)
  fig = Figure()
  ax = Axis(fig[1, 1], xlabel=L"R", ylabel=L"E")
  lines!(ax, R_vec, F.(R_vec))
  save(joinpath(RESULTS_FOLDER, "outer-optimisation.pdf"), fig)
  return fig
end

function plotAnalyticSolution()
  fig = Figure()
  alpha, beta = knownAnalyticParams.alpha, knownAnalyticParams.beta
  env = Utils.createEnvironment(knownAnalyticParams)
  R, analytic = AnalyticSolutions.explicitSolution(x_vec_noends; p=knownAnalyticParams)
  ax = Axis(fig[1, 1], title=L"\text{Analytic Solution with } \alpha=%$alpha \text{ and } \beta=%$beta", xlabel=L"x", ylabel=L"\rho(x)")
  lines!(ax, x_vec_noends, Utils.rho(x_vec_noends, Solver.solve(4, R, env), env), label=L"N = 4")
  lines!(ax, x_vec_noends, Utils.rho(x_vec_noends, Solver.solve(8, R, env), env), label=L"N = 8")
  lines!(ax, x_vec_noends, Utils.rho(x_vec_noends, Solver.solve(20, R, env), env), label=L"N = 20")
  lines!(ax, x_vec_noends, analytic, linewidth=3.0, linestyle=:dash, label="Analytic")
  axislegend(ax)
  ax = Axis(fig[2, 1], title="Squared Error", yscale=log10, xlabel=L"x", ylabel=L"\rho_N(x) - \rho(x)")
  lines!(ax, x_vec_noends, abs.(Utils.rho(x_vec_noends, Solver.solve(4, R, env), env) .- analytic), label=L"N = 4")
  lines!(ax, x_vec_noends, abs.(Utils.rho(x_vec_noends, Solver.solve(8, R, env), env) .- analytic), label=L"N = 8")
  lines!(ax, x_vec_noends, abs.(Utils.rho(x_vec_noends, Solver.solve(20, R, env), env) .- analytic), label=L"N = 20")
  axislegend(ax)
  save(joinpath(RESULTS_FOLDER, "analytic-solution.pdf"), fig)
  return fig
end

function plotConvergenceToAnalytic()
  Ns = 1:1:34
  env = Utils.createEnvironment(knownAnalyticParams)
  errors = zeros(length(Ns))
  R, analytic = AnalyticSolutions.explicitSolution(r_vec[1:end-1]; p=knownAnalyticParams)
  for k in eachindex(Ns)
    N = Ns[k]
    solution = Solver.solve(N, R, env)
    this = Utils.rho(r_vec[1:end-1], solution, env)
    errors[k] = sum((this - analytic) .^ 2) / length(r_vec)
  end

  fig = Figure()
  ax = Axis(fig[1, 1], yscale=log10, xlabel=L"N", ylabel="Squared Error")
  lines!(ax, Ns, errors)
  scatter!(ax, Ns, errors, color=:red)
  save(joinpath(RESULTS_FOLDER, "convergence-to-analytic.pdf"), fig)
  return fig
end

function plotAll()
  plotDifferentOrderSolutions()
  plotOperators()
  plotConvergence()
  plotSpatialEnergyDependence()
  plotOuterOptimisation()
  plotAnalyticSolution()
  return
end

println("All plotted for today?")
