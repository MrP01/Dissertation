using CairoMakie
using SparseArrays
using LaTeXStrings
import CSV
import DataFrames

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

function saveFig(fig::Figure, name::String)
  save(joinpath(RESULTS_FOLDER, "$name.pdf"), fig)
  @info "Exported $name.pdf"
end
macro LT_str(s::String)
  return latexstring(raw"\text{" * s * "}")
end

function plotDifferentOrderSolutions()
  fig = Figure()
  alpha, beta, d = defaultParams.alpha, defaultParams.beta, defaultParams.d
  ax = Axis(fig[1, 1], xlabel=L"\text{Radial Position}~x", ylabel=L"\text{Probability Density}~\rho(|x|)",
    title=L"\text{Solutions of different order}~N~\text{with}~(\alpha, \beta, d) = (%$alpha, %$beta, %$d)")
  R = defaultParams.R0
  env = Utils.defaultEnv
  # lines!(ax, x_vec_noends, obtainMeasure(x_vec_noends, 2), label="N = 2")
  lines!(ax, x_vec_noends, Utils.rho(x_vec_noends, Solver.solve(3, R, env), env), label="N = 3")
  lines!(ax, x_vec_noends, Utils.rho(x_vec_noends, Solver.solve(4, R, env), env), label="N = 4")
  lines!(ax, x_vec_noends, Utils.rho(x_vec_noends, Solver.solve(5, R, env), env), label="N = 5")
  lines!(ax, x_vec_noends, Utils.rho(x_vec_noends, Solver.solve(6, R, env), env), label="N = 6")
  lines!(ax, x_vec_noends, Utils.rho(x_vec_noends, Solver.solve(7, R, env), env), label="N = 7")
  axislegend(ax)
  saveFig(fig, "solution-increasing-order")
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
  saveFig(fig, "attractive-repulsive-operators")
  return fig
end

function plotSpatialEnergyDependence()
  fig = Figure()
  alpha, beta, d = defaultParams.alpha, defaultParams.beta, defaultParams.d
  ax = Axis(fig[1, 1], xlabel=L"\text{Radial Distance}~r", ylabel=L"\text{Energy}~E(r)",
    title=L"\text{Energy Dependence on}~r~\text{with}~(\alpha, \beta, d) = (%$alpha, %$beta, %$d)")
  for R in 0.4:0.2:1.2
    solution = Solver.solve(12, R, Utils.defaultEnv)
    lines!(ax, r_vec_noend, Utils.totalEnergy(solution, R, r_vec_noend, Utils.defaultEnv), label=L"R = %$R")
  end
  axislegend(ax)
  saveFig(fig, "energy-dependence-on-r")
  return fig
end

function plotStepByStepConvergence()
  Ns = 1:1:22
  env = Utils.defaultEnv
  errors = zeros(length(Ns))
  R = Utils.defaultParams.R0
  best = Utils.rho(r_vec_noend, Solver.solve(24, R, env), env)
  for k in eachindex(Ns)
    N = Ns[k]
    solution = Solver.solve(N, R, env)
    this = Utils.rho(r_vec_noend, solution, env)
    errors[k] = sum((this - best) .^ 2) / length(r_vec)
  end

  fig = Figure()
  ax = Axis(fig[1, 1], yscale=log10, xlabel=L"N", ylabel=LT"Squared Error", title=LT"Step by Step Convergence")
  lines!(ax, Ns, errors)
  scatter!(ax, Ns, errors, color=:red)
  saveFig(fig, "convergence")
  return fig
end

function plotOuterOptimisation()
  R_vec = 0.25:0.02:1.5
  env = Utils.defaultEnv
  F(R) = Utils.totalEnergy(Solver.solve(8, R, env), R, 0.0, env)
  fig = Figure()
  alpha, beta, d = round(knownAnalyticParams.alpha, digits=2), round(knownAnalyticParams.beta, digits=2), knownAnalyticParams.d
  ax = Axis(fig[1, 1], xlabel=L"R", ylabel=L"E(R)", title=L"\text{Energy Optimisation with}~(\alpha, \beta, d) = (%$alpha, %$beta, %$d)")
  lines!(ax, R_vec, F.(R_vec))
  saveFig(fig, "outer-optimisation")
  return fig
end

function plotAnalyticSolution()
  fig = Figure()
  alpha, beta, d = round(knownAnalyticParams.alpha, digits=2), round(knownAnalyticParams.beta, digits=2), knownAnalyticParams.d
  env = Utils.createEnvironment(knownAnalyticParams)
  R, analytic = AnalyticSolutions.explicitSolution(x_vec_noends; p=knownAnalyticParams)
  ax = Axis(fig[1, 1], title=L"\text{Analytic Solution with}~(\alpha, \beta, d) = (%$alpha, %$beta, %$d)", xlabel=L"x", ylabel=L"\rho(x)")
  lines!(ax, x_vec_noends, Utils.rho(x_vec_noends, Solver.solve(4, R, env), env), label=L"N = 4")
  lines!(ax, x_vec_noends, Utils.rho(x_vec_noends, Solver.solve(8, R, env), env), label=L"N = 8")
  lines!(ax, x_vec_noends, Utils.rho(x_vec_noends, Solver.solve(20, R, env), env), label=L"N = 20")
  lines!(ax, x_vec_noends, analytic, linewidth=4.0, linestyle=:dash, label=LT"Analytic")
  axislegend(ax)
  ax = Axis(fig[2, 1], title=LT"Absolute Error", yscale=log10, xlabel=L"x", ylabel=L"|\rho_N(x) - \rho(x)|")
  lines!(ax, x_vec_noends, abs.(Utils.rho(x_vec_noends, Solver.solve(4, R, env), env) .- analytic), label=L"N = 4")
  lines!(ax, x_vec_noends, abs.(Utils.rho(x_vec_noends, Solver.solve(8, R, env), env) .- analytic), label=L"N = 8")
  lines!(ax, x_vec_noends, abs.(Utils.rho(x_vec_noends, Solver.solve(20, R, env), env) .- analytic), label=L"N = 20")
  axislegend(ax)
  saveFig(fig, "analytic-solution")
  return fig
end

function plotConvergenceToAnalytic()
  Ns = 1:1:34
  env = Utils.createEnvironment(knownAnalyticParams)
  errors = zeros(length(Ns))
  R, analytic = AnalyticSolutions.explicitSolution(r_vec_noend; p=knownAnalyticParams)
  for k in eachindex(Ns)
    N = Ns[k]
    solution = Solver.solve(N, R, env)
    this = Utils.rho(r_vec_noend, solution, env)
    errors[k] = sum((this - analytic) .^ 2) / length(r_vec)
  end

  fig = Figure()
  alpha, beta, d = round(knownAnalyticParams.alpha, digits=2), round(knownAnalyticParams.beta, digits=2), knownAnalyticParams.d
  ax = Axis(fig[1, 1], yscale=log10, xlabel=L"N", ylabel=LT"Squared Error",
    title=L"\text{Convergence to analytic solution with}~(\alpha, \beta, d) = (%$alpha, %$beta, %$d)")
  lines!(ax, Ns, errors)
  scatter!(ax, Ns, errors, color=:red)
  saveFig(fig, "convergence-to-analytic")
  return fig
end

function plotParameterVariations()
  fig = Figure(resolution=(650, 900))
  R = 0.8
  p = Parameters()
  alpha, beta, d = round(p.alpha, digits=2), round(p.beta, digits=2), p.d
  ax = Axis(fig[1, 1], ylabel=L"\rho(r)", title=L"\text{Varying}~\alpha~\text{with}~(\beta, d) = (%$beta, %$d)")
  for alpha in 0.3:0.6:2.7
    env = Utils.createEnvironment(Parameters(alpha=alpha))
    lines!(ax, r_vec_noend, Utils.rho(r_vec_noend, Solver.solve(8, R, env), env), label=L"\alpha = %$alpha")
  end
  axislegend(ax, position=:lt)
  ax = Axis(fig[2, 1], ylabel=L"\rho(r)", title=L"\text{Varying}~\beta~\text{with}~(\alpha, d) = (%$alpha, %$d)")
  for beta in 0.3:0.6:2.7
    env = Utils.createEnvironment(Parameters(beta=beta))
    lines!(ax, r_vec_noend, Utils.rho(r_vec_noend, Solver.solve(8, R, env), env), label=L"\beta = %$beta")
  end
  axislegend(ax, position=:lt)
  ax = Axis(fig[3, 1], ylabel=L"\rho(r)", title=L"\text{Varying}~m~\text{with}~(\alpha, \beta, d) = (%$alpha, %$beta, %$d)")
  for m in 1:4
    env = Utils.createEnvironment(Parameters(m=m))
    lines!(ax, r_vec_noend, Utils.rho(r_vec_noend, Solver.solve(8, R, env), env), label=L"m = %$m")
  end
  axislegend(ax, position=:lt)
  ax = Axis(fig[4, 1], xlabel=L"r", ylabel=L"\rho(r)", title=L"\text{Varying}~d~\text{with}~(\alpha, \beta) = (%$alpha, %$beta)")
  for d in 1:5
    env = Utils.createEnvironment(Parameters(d=d, m=3))
    lines!(ax, r_vec_noend, Utils.rho(r_vec_noend, Solver.solve(8, R, env), env), label=L"d = %$d")
  end
  axislegend(ax, position=:lt)
  saveFig(fig, "varying-parameters")
  return fig
end

function plotSimulationHistogram()
  fig = Figure()
  alpha, beta, d = defaultParams.alpha, defaultParams.beta, defaultParams.d
  df = CSV.read("/tmp/position-histogram.csv", DataFrames.DataFrame, header=["hist"])
  ax = Axis(fig[1, 1], xlabel=L"\text{Position}~x", ylabel=LT"Density",
    title=L"\text{Particle Simulation Output Distribution}~(\alpha, \beta, d) = (%$alpha, %$beta, %$d)")
  barplot!(ax, df.hist)
  saveFig(fig, "simulation-histogram")
  return fig
end

function plotJacobiConvergence()
  P = Utils.defaultEnv.P
  f(x) = exp(x^2) # function we want to expand
  f_N = P[:, 1:10] \ f.(axes(P, 1))
  fig = Figure()
  ax = Axis(fig[1, 1], yscale=log10, xlabel=L"x", ylabel=LT"Absolute Error",
    title=L"\text{Expansion of the function}~f(x) = \exp(x^2)")
  for k in 2:10
    lines!(ax, r_vec, vec(abs.(sum(f_N[1:k] .* P[r_vec, 1:k]', dims=1) - f.(r_vec)')), label=L"N = %$k")
  end
  saveFig(fig, "jacobi-expansions")
  return fig
end

function plotAll()
  plotDifferentOrderSolutions()
  plotOperators()
  plotStepByStepConvergence()
  plotSpatialEnergyDependence()
  plotOuterOptimisation()
  plotAnalyticSolution()
  plotConvergenceToAnalytic()
  return
end

println("All plotted for today?")
