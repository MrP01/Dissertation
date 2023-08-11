using CairoMakie
using SparseArrays
using LaTeXStrings
import CSV
import DataFrames

include("./parameters.jl")
include("./solver.jl")
include("./analyticsolutions.jl")
import .Params
import .Utils
import .Solver
import .AttractiveRepulsiveSolver
import .GeneralKernelSolver
import .AnalyticSolutions

const RESULTS_FOLDER = joinpath(@__DIR__, "..", "figures", "results")

r_vec = 0:0.005:1
r_vec_noend = r_vec[1:end-1]
x_vec = -1:0.002:1
x_vec_noends = x_vec[2:end-1]
dissColours = Makie.wong_colors()
pow10tickformat(values) = [L"10^{%$(Int64(round(value)))}" for value in values]

function runSimulator(p::Params.Parameters, iterations=2000)
  if isa(p.potential, Params.AttractiveRepulsive)
    mode = "attrep"
    potentialParams = [p.potential.alpha, p.potential.beta]
  elseif isa(p.potential, Params.MorsePotential)
    mode = "morse"
    potentialParams = [p.potential.C_att, p.potential.l_att, p.potential.C_rep, p.potential.l_rep]
  else
    error("Unkown potential")
  end
  cmd = Cmd(string.(["./build/simulator/experiments$(p.d)d", mode, p.d, iterations, potentialParams...]))
  @show cmd
  println("----- Running simulation -----")
  run(cmd)
  println("----- Simulation done -----")
end
function loadSimulatorData()
  posidf = CSV.read("/tmp/positions.csv", DataFrames.DataFrame, header=false)
  velodf = CSV.read("/tmp/velocities.csv", DataFrames.DataFrame, header=false)
  dimension = length(axes(posidf, 2))
  return posidf, velodf, dimension
end
function saveFig(fig::Figure, name::String)
  save(joinpath(RESULTS_FOLDER, "$name.pdf"), fig)
  @info "Exported $name.pdf"
end
macro LT_str(s::String)
  return latexstring(raw"\text{" * s * "}")
end

function plotDifferentOrderSolutions(p=Params.defaultParams)
  fig = Figure()
  env = Utils.createEnvironment(p)
  ax = Axis(fig[1, 1], xlabel=L"\text{Radial Position}~x", ylabel=L"\text{Probability Density}~\rho(|x|)",
    title=L"\text{Solutions of different order}~N~\text{with}~%$(Params.potentialParamsToLatex(p.potential)),~d=%$(p.d)")
  # lines!(ax, x_vec_noends, obtainMeasure(x_vec_noends, 2), label="N = 2")
  lines!(ax, x_vec_noends, Utils.rho(x_vec_noends, Solver.solve(3, p.R0, env), env), label="N = 3")
  lines!(ax, x_vec_noends, Utils.rho(x_vec_noends, Solver.solve(4, p.R0, env), env), label="N = 4")
  lines!(ax, x_vec_noends, Utils.rho(x_vec_noends, Solver.solve(5, p.R0, env), env), label="N = 5")
  lines!(ax, x_vec_noends, Utils.rho(x_vec_noends, Solver.solve(6, p.R0, env), env), label="N = 6")
  lines!(ax, x_vec_noends, Utils.rho(x_vec_noends, Solver.solve(7, p.R0, env), env), label="N = 7")
  axislegend(ax)
  saveFig(fig, "solution-increasing-order")
  return fig
end

function plotGeneralSolutionApproximation(p=Params.morsePotiParams)
  fig = Figure()
  ax = Axis(fig[1, 1], xlabel=L"\text{Radial Position}~x", ylabel=L"\text{Probability Density}~\rho(|x|)",
    title=L"\text{Monomial Terms}")
  for M in 2:6
    env = Utils.createEnvironment(p, M)
    # env = Utils.createEnvironment(Params.Parameters(d=p.d, m=p.m, R0=p.R0, potential=p.potential, friction=p.friction, M=M))
    lines!(ax, x_vec_noends, Utils.rho(x_vec_noends, Solver.solve(8, p.R0, env), env), label="M = $M")
  end
  axislegend(ax)
  saveFig(fig, "morse-solutions")
  return fig
end

function plotAttRepOperators(N=30)
  p = Params.defaultParams
  env = Utils.createEnvironment(p)
  alpha, beta = p.potential.alpha, p.potential.beta
  op1 = AttractiveRepulsiveSolver.constructOperator(N, alpha, env)
  op2 = AttractiveRepulsiveSolver.constructOperator(N, beta, env)
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

function plotFullOperator(N=30, p=Params.morsePotiParams)
  fig = Figure(resolution=(600, 500))
  env = Utils.createEnvironment(p)
  op1 = Solver.constructOperatorFromEnv(N, p.R0, env)
  ax = Axis(fig[1, 1][1, 1], yreversed=true, title=L"\text{Operator}")
  s = spy!(ax, sparse(log10.(abs.(op1))), marker=:rect, markersize=32, framesize=0)
  Colorbar(fig[1, 1][1, 2], s, flipaxis=false, tickformat=pow10tickformat)
  saveFig(fig, "morse-operator")
  return fig
end

function plotSpatialEnergyDependence()
  fig = Figure()
  p = Params.defaultParams
  alpha, beta, d = p.potential.alpha, p.potential.beta, p.d
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
  R = env.p.R0
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

function plotMonomialBasisConvergence(p=Params.morsePotiParams)
  Ms = 1:10
  R = p.R0
  errors = zeros(length(Ms))
  env = Utils.createEnvironment(p, 12)
  best = Utils.rho(r_vec_noend, Solver.solve(24, R, env), env)
  for k in eachindex(Ms)
    M = Ms[k]
    env = Utils.createEnvironment(p, M)
    solution = Solver.solve(24, R, env)
    this = Utils.rho(r_vec_noend, solution, env)
    errors[k] = sum((this - best) .^ 2) / length(r_vec)
  end

  fig = Figure()
  ax = Axis(fig[1, 1], yscale=log10, xlabel=L"M", ylabel=LT"Squared Error", title=LT"Step by Step Convergence")
  lines!(ax, Ms, errors)
  scatter!(ax, Ms, errors, color=:red)
  saveFig(fig, "monomial-basis-convergence")
  return fig
end

function plotOuterOptimisation(p=Params.knownAnalyticParams)
  R_vec = 0.25:0.02:1.5
  env = Utils.createEnvironment(p)
  F(R) = Utils.totalEnergy(Solver.solve(8, R, env), R, 0.0, env)
  fig = Figure()
  ax = Axis(fig[1, 1], xlabel=L"R", ylabel=L"E(R)",
    title=L"\text{Energy Optimisation with}~%$(Params.potentialParamsToLatex(p.potential)),~d=%$(p.d)")
  lines!(ax, R_vec, F.(R_vec))
  saveFig(fig, "outer-optimisation")
  return fig
end

function plotAnalyticSolution()
  fig = Figure()
  p = Params.knownAnalyticParams
  alpha, beta, d = round(p.potential.alpha, digits=2), round(p.potential.beta, digits=2), p.d
  env = Utils.createEnvironment(p)
  R, analytic = AnalyticSolutions.explicitSolution(x_vec_noends; p=p)
  ax = Axis(fig[1, 1], title=L"\text{Analytic Solution with}~", xlabel=L"x", ylabel=L"\rho(x)")
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
  env = Utils.createEnvironment(Params.knownAnalyticParams)
  errors = zeros(length(Ns))
  R, analytic = AnalyticSolutions.explicitSolution(r_vec_noend; p=env.p)
  for k in eachindex(Ns)
    N = Ns[k]
    solution = Solver.solve(N, R, env)
    this = Utils.rho(r_vec_noend, solution, env)
    errors[k] = sum((this - analytic) .^ 2) / length(r_vec)
  end

  fig = Figure()
  alpha, beta, d = round(env.p.potential.alpha, digits=2), round(env.p.potential.beta, digits=2), env.p.d
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
  p = Params.Parameters()
  alpha, beta, d = round(p.potential.alpha, digits=2), round(p.potential.beta, digits=2), p.d
  ax = Axis(fig[1, 1], ylabel=L"\rho(r)", title=L"\text{Varying}~\alpha~\text{with}~(\beta, d) = (%$beta, %$d)")
  for alpha in 0.3:0.6:2.7
    env = Utils.createEnvironment(Params.Parameters(potential=Params.AttractiveRepulsive(alpha=alpha)))
    lines!(ax, r_vec_noend, Utils.rho(r_vec_noend, Solver.solve(8, R, env), env), label=L"\alpha = %$alpha")
  end
  axislegend(ax, position=:lt)
  ax = Axis(fig[2, 1], ylabel=L"\rho(r)", title=L"\text{Varying}~\beta~\text{with}~(\alpha, d) = (%$alpha, %$d)")
  for beta in 0.3:0.6:2.7
    env = Utils.createEnvironment(Params.Parameters(potential=Params.AttractiveRepulsive(beta=beta)))
    lines!(ax, r_vec_noend, Utils.rho(r_vec_noend, Solver.solve(8, R, env), env), label=L"\beta = %$beta")
  end
  axislegend(ax, position=:lt)
  ax = Axis(fig[3, 1], ylabel=L"\rho(r)", title=L"\text{Varying}~m~\text{with}~(\alpha, \beta, d) = (%$alpha, %$beta, %$d)")
  for m in 1:4
    env = Utils.createEnvironment(Params.Parameters(m=m))
    lines!(ax, r_vec_noend, Utils.rho(r_vec_noend, Solver.solve(8, R, env), env), label=L"m = %$m")
  end
  axislegend(ax, position=:lt)
  ax = Axis(fig[4, 1], xlabel=L"r", ylabel=L"\rho(r)", title=L"\text{Varying}~d~\text{with}~(\alpha, \beta) = (%$alpha, %$beta)")
  for d in 1:5
    env = Utils.createEnvironment(Params.Parameters(d=d, m=3))
    lines!(ax, r_vec_noend, Utils.rho(r_vec_noend, Solver.solve(8, R, env), env), label=L"d = %$d")
  end
  axislegend(ax, position=:lt)
  saveFig(fig, "varying-parameters")
  return fig
end

function plotSimulationHistograms()
  fig = Figure()
  df = CSV.read("/tmp/positions.csv", DataFrames.DataFrame, header=false)
  dimension = length(axes(df, 2))
  center = [sum(df[!, k]) / length(df[!, k]) for k in 1:dimension]
  @show center
  radialDistance = hypot.([df[!, k] .- center[k] for k in 1:dimension]...)
  ax = Axis(fig[1, 1], xlabel=L"\text{Radial distance}~r", ylabel=LT"Density",
    title=L"\text{Particle Simulation Output Distribution}")
  hist!(ax, radialDistance, bins=0:0.02:(maximum(radialDistance)*1.05), color=dissColours[1])

  df = CSV.read("/tmp/position-histogram.csv", DataFrames.DataFrame, header=["hist"])
  ax = Axis(fig[2, 1], xlabel=L"\text{Radial distance}~r", ylabel=LT"Density",
    title=L"\text{Averaged Simulation Output Distribution over 20 runs}")
  barplot!(ax, df.hist, gap=0, color=dissColours[2])

  df = CSV.read("/tmp/velocities.csv", DataFrames.DataFrame, header=false)
  velocity = hypot.([df[!, k] for k in 1:dimension]...)
  ax = Axis(fig[3, 1], xlabel=L"\text{Velocity}~v", ylabel=LT"Density",
    title=L"\text{Simulation Output Velocity Distribution}")
  hist!(ax, velocity, bins=20, color=dissColours[3])

  saveFig(fig, "simulation-histogram")
  return fig
end

function plotSimulationAndSolverComparison(p::Params.Parameters=Params.known2dParams)
  env = Utils.createEnvironment(p)
  runSimulator(env.p)

  df = CSV.read("/tmp/positions.csv", DataFrames.DataFrame, header=false)
  dimension = length(axes(df, 2))
  @show center = [sum(df[!, k]) / length(df[!, k]) for k in 1:dimension]
  radialDistance = hypot.([df[!, k] .- center[k] for k in 1:dimension]...)
  maxR = maximum(radialDistance)
  solution = Utils.rho(r_vec_noend, Solver.solve(12, maxR, env), env)
  r = r_vec_noend * maxR

  fig = Figure()
  ax = Axis(fig[1, 1], xlabel=L"\text{Radial distance}~r", ylabel=LT"Density",
    title=L"\text{Particle Simulation Output Distribution}")
  hist!(ax, radialDistance, bins=0:0.06:(maximum(radialDistance)*1.05), scale_to=maximum(solution),
    label=LT"Particle Simulation")
  lines!(ax, r, solution, color=:red, linewidth=3.0, label=LT"Spectral Method")
  ylims!(ax, 0, maximum(solution) * 1.05)
  axislegend(ax, position=:lt)
  saveFig(fig, "simulation-solver-comparison")
  return fig
end

function plotSimulationQuiver(p::Union{Params.Parameters,Symbol}=:nothing, iterations::Int64=2000)
  if p != :nothing
    runSimulator(p, iterations)
  end
  posidf, velodf, dimension = loadSimulatorData()
  @assert dimension >= 2

  fig = Figure()
  ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y", title=L"\text{Simulation Output}~(d = %$dimension)")
  scatter!(ax, posidf[!, 1], posidf[!, 2], color=dissColours[1])
  quiver!(ax, posidf[!, 1], posidf[!, 2], velodf[!, 1] / 20, velodf[!, 2] / 20,
    color=dissColours[4], linewidth=2)
  saveFig(fig, "simulation-quiver")
  return fig
end

function plotPhaseSpace()
  fig = Figure()
  posidf, velodf, dimension = loadSimulatorData()
  ax = Axis(fig[1, 1], xlabel=L"\text{First coordinate}~x", ylabel=L"\text{First velocity component}~v_x",
    title=L"\text{Simulation Output Phase Space}~(d = %$dimension)")
  scatter!(ax, posidf[!, 1], velodf[!, 1])
  ax = Axis(fig[2, 1], xlabel=L"\text{Radial distance}~r", ylabel=L"\text{Velocity}~v",
    title=L"\text{Simulation Output Phase Space}~(d = %$dimension)")
  radialDistance = hypot.([posidf[!, k] for k in 1:dimension]...)
  velocity = hypot.([velodf[!, k] for k in 1:dimension]...)
  scatter!(ax, radialDistance, velocity)
  saveFig(fig, "phase-space-plot")
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
  plotAttRepOperators()
  plotOuterOptimisation()
  plotAnalyticSolution()
  plotConvergenceToAnalytic()
  plotJacobiConvergence()
  plotParameterVariations()
  plotPhaseSpace()
  plotSimulationAndSolverComparison()
  plotSimulationHistograms()
  plotSimulationQuiver()
  plotStepByStepConvergence()
  plotSpatialEnergyDependence()
  return
end

println("All plotted for today?")
