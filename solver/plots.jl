using CairoMakie
using SparseArrays
using LaTeXStrings
import CSV
import DataFrames
import Optim
import SpecialFunctions

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
_extra_pdf = true

r_vec = 0:0.005:1
r_vec_noend = r_vec[1:end-1]
x_vec = -1:0.002:1
x_vec_noends = x_vec[2:end-1]
x_vec_noends2 = x_vec[3:end-2]
x_vec_noends3 = x_vec[4:end-3]
pow10tickformat(values) = [L"10^{%$(Int64(round(value)))}" for value in values]

dissertationColours = Makie.wong_colors()
dissertationColours[1] = RGBAf(39, 63, 111, 255) / 255  # lighterOxfordBlue
dissertationColours[2] = RGBAf(175, 148, 72, 255) / 255  # oxfordGolden
dissertationColours[3] = RGBAf(101, 145, 87, 255) / 255  # wongGreen
dissertationColours[4] = RGBAf(204, 121, 167, 255) / 255  # wongPurple
dissertationColours[5] = RGBAf(158, 179, 194, 255) / 255  # coolGray
# dissertationColourmap = [dissertationColours[1], dissertationColours[2]]
dissertationColourmap = :viridis
dissertationTheme = Theme(palette=(color=dissertationColours,), fontsize=20)
set_theme!(dissertationTheme)

function runSimulator(p::Params.Parameters, iterations=2000, big=true)
  if isa(p.potential, Params.AttractiveRepulsive)
    mode = "attrep"
    potentialParams = [p.potential.alpha, p.potential.beta]
  elseif isa(p.potential, Params.MorsePotential)
    mode = "morse"
    potentialParams = [p.potential.C_att, p.potential.l_att, p.potential.C_rep, p.potential.l_rep]
  elseif isa(p.potential, Params.MixedPotential)
    mode = "mixed"
    potentialParams = [p.potential.morseC, p.potential.morsel, p.potential.attrepPower]
  elseif isa(p.potential, Params.AbsoluteValuePotential)
    mode = "absvalue"
    potentialParams = []
  else
    error("Unkown potential")
  end
  boxScaling = ceil(p.R0)
  bigstring = big ? "big" : ""
  cmd = Cmd(string.(["./build/simulator/experiments$(p.d)d$(bigstring)",
    mode, p.d, iterations, boxScaling, p.friction.selfPropulsion, p.friction.frictionCoeff, potentialParams...]))
  @show cmd
  println("----- Running simulation with $(p.name) -----")
  run(cmd)
  println("----- Simulation done -----")
end
function loadSimulatorData()
  posidf = CSV.read("/tmp/positions.csv", DataFrames.DataFrame, header=false)
  velodf = CSV.read("/tmp/velocities.csv", DataFrames.DataFrame, header=false)
  dimension = length(axes(posidf, 2))
  return posidf, velodf, dimension
end
function _saveFigCommon(name::String)
  if _extra_pdf
    name = name * ".extra"
  end
  if Makie.current_backend() != CairoMakie
    return
  end
  return name
end
function saveFig(fig::Figure, name::String)
  name = _saveFigCommon(name)
  if isnothing(name)
    return
  end
  path = joinpath(RESULTS_FOLDER, name)
  save("$path.pdf", fig)
  @info "Exported $name.pdf"
end
function saveFig(fig::Figure, name::String, p::Params.Parameters; throughEps=false)
  name = _saveFigCommon(name)
  if isnothing(name)
    return
  end
  path = joinpath(RESULTS_FOLDER, p.name, name)
  if ~isdir(joinpath(RESULTS_FOLDER, p.name))
    mkdir(joinpath(RESULTS_FOLDER, p.name))
  end
  if throughEps
    save("$path.eps", fig)
    run(`epstopdf $path.eps -o $path.pdf`)
  else
    save("$path.pdf", fig)
  end
  @info "Exported $(p.name)/$name.pdf"
end
function p2tex(p::Params.Parameters)
  return Params.potentialParamsToLatex(p.potential, true) * ",~d=$(p.d)"
end
function p2tex(p::Params.Parameters, R::Float64)
  return Params.potentialParamsToLatex(p.potential, true) * ",~d=$(p.d),~R_{\\text{opt}}\\approx$(round(R, digits=2))"
end
macro LT_str(s::String)
  return latexstring(raw"\text{" * s * "}")
end

function plotDifferentOrderSolutions(p=Params.defaultParams)
  fig = Figure(resolution=(900, 450))
  env = Utils.createEnvironment(p)
  @show R_opt = Solver.outerOptimisation(18, env).minimizer[1]
  ax = Axis(fig[1, 1], xlabel=L"\text{Radial Position}~x", ylabel=L"\text{Probability Density}~\rho(|x|)",
    title=L"\text{Solutions of different order}~N~\text{with}~%$(p2tex(p, R_opt))")
  # lines!(ax, x_vec_noends, obtainMeasure(x_vec_noends, 2), label="N = 2")
  lines!(ax, x_vec_noends2, Utils.rho(x_vec_noends2, Solver.solve(4, R_opt, env), env), label="N = 4")
  lines!(ax, x_vec_noends2, Utils.rho(x_vec_noends2, Solver.solve(6, R_opt, env), env), label="N = 6")
  lines!(ax, x_vec_noends2, Utils.rho(x_vec_noends2, Solver.solve(12, R_opt, env), env), label="N = 12")
  lines!(ax, x_vec_noends2, Utils.rho(x_vec_noends2, Solver.solve(42, R_opt, env), env), label="N = 42", linewidth=2)
  axislegend(ax)
  saveFig(fig, "solution-increasing-order", p)
  return fig
end

function plotGeneralSolutionApproximation(p=Params.morsePotiParams)
  env = Utils.createEnvironment(p, 8)
  fig = Figure(resolution=(900, 450))
  @show R_opt = Solver.outerOptimisation(19, env).minimizer[1]
  ax = Axis(fig[1, 1], xlabel=L"\text{Radial position}~x", ylabel=L"\text{Probability density}~\rho(|x|)",
    title=L"\text{General Kernel Solution with}~G~\text{Monomial Terms}~%$(p2tex(p, R_opt))")
  for G in 6:2:12
    env = Utils.createEnvironment(p, G)
    solution = Solver.solveWithRegularisation(19, R_opt, env, p.s0)
    lines!(ax, x_vec_noends2, Utils.rho(x_vec_noends2, solution, env), label="G = $G")
  end
  axislegend(ax)
  saveFig(fig, "monomial-solutions", p)
  return fig
end

function plotAttRepOperators(p=Params.defaultParams; N=60)
  @assert isa(p.potential, Params.AttractiveRepulsive)
  env = Utils.createEnvironment(p)
  alpha, beta = p.potential.alpha, p.potential.beta
  op1 = AttractiveRepulsiveSolver.recursivelyConstructOperator(N, alpha, env)
  op2 = AttractiveRepulsiveSolver.recursivelyConstructOperator(N, beta, env)
  fig = Figure(resolution=(920, 400))
  ax = Axis(fig[1, 1][1, 1], yreversed=true, title=L"\text{Attractive Operator}~(\alpha = %$alpha)", xlabel=LT"Column Index", ylabel=LT"Row Index")
  s = spy!(ax, sparse(log10.(abs.(op1))), marker=:rect, markersize=12, framesize=0, colormap=dissertationColourmap)
  Colorbar(fig[1, 1][1, 2], s, flipaxis=false, tickformat=pow10tickformat)
  ax = Axis(fig[1, 2][1, 1], yreversed=true, title=L"\text{Repulsive Operator}~(\beta = %$beta)", xlabel=LT"Column Index", ylabel=LT"Row Index")
  s = spy!(ax, sparse(log10.(abs.(op2))), marker=:rect, markersize=12, framesize=0, colormap=dissertationColourmap)
  Colorbar(fig[1, 2][1, 2], s, flipaxis=false, tickformat=pow10tickformat)
  saveFig(fig, "attractive-repulsive-operators", p)
  return fig
end

function enhancedSpyPlot(matrix, title="")
  @debug Utils.opCond(matrix)
  fig = Figure(resolution=(600, 500))
  ax = Axis(fig[1, 1][1, 1], yreversed=true, title=title, xlabel=LT"Column Index", ylabel=LT"Row Index")
  s = spy!(ax, sparse(log10.(abs.(matrix))), marker=:rect, markersize=12, framesize=0, colormap=dissertationColourmap)
  Colorbar(fig[1, 1][1, 2], s, flipaxis=false, tickformat=pow10tickformat)
  return fig
end

function plotFullOperator(p=Params.morsePotiParams; N=60)
  env = Utils.createEnvironment(p)
  op1 = Solver.constructOperatorFromEnv(N, p.R0, env)
  fig = enhancedSpyPlot(op1, L"\text{Full Operator}~%$(p2tex(p))")
  saveFig(fig, "full-operator", p)
  return fig
end

function plotSpatialEnergyDependence(p=Params.defaultParams; N=12)
  fig = Figure(resolution=(800, 450))
  ax = Axis(fig[1, 1], xlabel=L"\text{Radial Distance}~r", ylabel=L"\text{Energy}~E(r)",
    title=L"\text{Energy Dependence on}~r~\text{with}~%$(p2tex(p))")
  for R in 0.4:0.2:1.2
    solution = Solver.solve(N, R, Utils.defaultEnv)
    lines!(ax, r_vec_noend, Utils.notTotalEnergy(solution, R, r_vec_noend, Utils.defaultEnv), label=L"R = %$R")
  end
  axislegend(ax)
  saveFig(fig, "energy-dependence-on-r", p)
  return fig
end

function plotStepByStepConvergence(p=Params.defaultParams)
  Ns = 1:1:18
  env = Utils.createEnvironment(p)
  errors = zeros(length(Ns))
  @show R_opt = Solver.outerOptimisation(19, env).minimizer[1]
  best = Utils.rho(r_vec_noend, Solver.solve(50, R_opt, env), env)
  for k in eachindex(Ns)
    N = Ns[k]
    solution = Solver.solve(N, R_opt, env)
    this = Utils.rho(r_vec_noend, solution, env)
    errors[k] = sum((this - best) .^ 2) / length(r_vec)
  end

  fig = Figure(resolution=(800, 450))
  # TODO: for all squared errors, give formula in the plot
  ax = Axis(fig[1, 1], yscale=log10, xlabel=L"N", ylabel=LT"Squared Error",
    title=L"\text{Step by Step Convergence with}~%$(p2tex(p))")
  lines!(ax, Ns, errors)
  scatter!(ax, Ns, errors, color=dissertationColours[4])
  saveFig(fig, "convergence", p)
  return fig
end

function plotMonomialBasisConvergence(p=Params.morsePotiParams)
  Gs = 2:11
  R = p.R0
  errors = zeros(length(Gs))
  env = Utils.createEnvironment(p, Gs[end] + 1)
  best = Utils.rho(r_vec_noend, Solver.solve(24, R, env), env)
  for k in eachindex(Gs)
    G = Gs[k]
    env = Utils.createEnvironment(p, G)
    solution = Solver.solve(24, R, env)
    this = Utils.rho(r_vec_noend, solution, env)
    errors[k] = sum((this - best) .^ 2) / length(r_vec)
  end

  fig = Figure(resolution=(800, 450))
  ax = Axis(fig[1, 1], yscale=log10, xlabel=L"G", ylabel=LT"Squared Error", title=LT"Step by Step Convergence")
  lines!(ax, Gs, errors)
  scatter!(ax, Gs, errors, color=dissertationColours[4])
  saveFig(fig, "monomial-basis-convergence", p)
  return fig
end

function plotOuterOptimisation(p=Params.knownAnalyticParams; N=8)
  R_vec = 0.25:0.02:1.5
  env = Utils.createEnvironment(p)
  @show R_opt = Solver.outerOptimisation(N, env).minimizer[1]
  F(R) = Utils.totalEnergy(Solver.solve(N, R, env), R, env)
  fig = Figure(resolution=(800, 400))
  ax = Axis(fig[1, 1], xlabel=L"R", ylabel=L"E(R)",
    title=L"\text{Energy Optimisation with}~%$(p2tex(p))")
  lines!(ax, R_vec, F.(R_vec), label=LT"Energy")
  scatter!(ax, R_opt, F(R_opt), color=dissertationColours[3], label=LT"Optimum")
  axislegend(ax)
  saveFig(fig, "outer-optimisation", p)
  return fig
end

function plotVaryingRSolutions(p=Params.morsePotiParams)
  env = Utils.createEnvironment(p)
  fig = Figure()
  ax = Axis(fig[1, 1], xlabel=L"r", ylabel=L"\rho(r)",
    title=L"\text{Energy with}~%$(p2tex(p))")
  for R in 0.2:0.4:1.4
    lines!(ax, r_vec_noend, Utils.rho(r_vec_noend, Solver.solve(6, R, env), env), label="R = $R")
  end
  if p == Params.morsePotiParams
    ylims!(ax, -0.5, 1)
  end
  axislegend(ax)
  saveFig(fig, "varying-R-solutions", p)
  return fig
end

function plotAnalyticSolution(p=Params.knownAnalyticParams)
  Ns = 2 .^ (1:7)
  fig = Figure(resolution=(800, 800))
  env = Utils.createEnvironment(p)
  R, analytic = AnalyticSolutions.explicitSolution(x_vec_noends; p=p)
  OptR(N) = abs(Solver.outerOptimisation(N - 1, env).minimizer[1] - R)
  OptRs = OptR.(Ns)
  ax = Axis(fig[1, 1], title=L"\text{Analytic Solution with}~%$(p2tex(p))", xlabel=L"x", ylabel=L"\rho(x)")
  lines!(ax, x_vec_noends, Utils.rho(x_vec_noends, Solver.solve(4, R, env), env), label=L"N = 4")
  lines!(ax, x_vec_noends, Utils.rho(x_vec_noends, Solver.solve(8, R, env), env), label=L"N = 8")
  lines!(ax, x_vec_noends, Utils.rho(x_vec_noends, Solver.solve(20, R, env), env), label=L"N = 20")
  lines!(ax, x_vec_noends, Utils.rho(x_vec_noends, Solver.solve(60, R, env), env), label=L"N = 60")
  lines!(ax, x_vec_noends, analytic, linewidth=4.0, linestyle=:dash, label=LT"Analytic")
  axislegend(ax)
  ax = Axis(fig[2, 1], title=LT"Absolute Error", yscale=log10, xlabel=L"x", ylabel=L"|\rho_N(x) - \rho(x)|")
  lines!(ax, x_vec_noends, abs.(Utils.rho(x_vec_noends, Solver.solve(4, R, env), env) .- analytic), label=L"N = 4")
  lines!(ax, x_vec_noends, abs.(Utils.rho(x_vec_noends, Solver.solve(8, R, env), env) .- analytic), label=L"N = 8")
  lines!(ax, x_vec_noends, abs.(Utils.rho(x_vec_noends, Solver.solve(20, R, env), env) .- analytic), label=L"N = 20")
  lines!(ax, x_vec_noends, abs.(Utils.rho(x_vec_noends, Solver.solve(60, R, env), env) .- analytic), label=L"N = 60")
  axislegend(ax)
  ax = Axis(fig[3, 1], title=LT"Optimal Support Radius Error", xscale=log10, yscale=log10, xlabel=L"N", ylabel=L"|R_N - R|",
    xticks=Ns, height=120)
  lines!(ax, Ns, OptRs)
  scatter!(ax, Ns, OptRs, color=dissertationColours[4])
  saveFig(fig, "analytic-solution", p)
  return fig
end

function plotConvergenceToAnalytic(p=Params.knownAnalyticParams)
  Ns = 1:1:34
  env = Utils.createEnvironment(p)
  errors = zeros(length(Ns))
  regerrors = zeros(length(Ns))
  regerrors2 = zeros(length(Ns))
  R, analytic = AnalyticSolutions.explicitSolution(r_vec_noend; p=env.p)
  for k in eachindex(Ns)
    N = Ns[k]
    solution = Solver.solve(N, R, env)
    this = Utils.rho(r_vec_noend, solution, env)
    thisreg = Utils.rho(r_vec_noend, Solver.solveWithRegularisation(N, R, env, 1e-9), env)
    thisreg2 = Utils.rho(r_vec_noend, Solver.solveWithRegularisation(N, R, env, 1e-5), env)
    errors[k] = sum((this - analytic) .^ 2) / length(r_vec)
    regerrors[k] = sum((thisreg - analytic) .^ 2) / length(r_vec)
    regerrors2[k] = sum((thisreg2 - analytic) .^ 2) / length(r_vec)
  end

  fig = Figure(resolution=(800, 450))
  ax = Axis(fig[1, 1], yscale=log10, xlabel=L"N", ylabel=LT"Squared Error",
    title=L"\text{Convergence to analytic solution with}~%$(p2tex(p))")
  lines!(ax, Ns, errors)
  lines!(ax, Ns, regerrors2, color=dissertationColours[2])
  lines!(ax, Ns, regerrors, color=dissertationColours[3])
  scatter!(ax, Ns, regerrors2, color=dissertationColours[2], label=L"\text{With regularisation}~(s=10^{-5})")
  scatter!(ax, Ns, regerrors, color=dissertationColours[3], label=L"\text{With regularisation}~(s=10^{-9})")
  scatter!(ax, Ns, errors, color=dissertationColours[4], label=LT"Without regularisation")
  axislegend(ax)
  saveFig(fig, "convergence-to-analytic", env.p)
  return fig
end

function plotDefaultParameterVariations()
  fig = Figure(resolution=(650, 900))
  R = 0.8
  p = Params.Parameters()
  alpha, beta, d = round(p.potential.alpha, digits=2), round(p.potential.beta, digits=2), p.d
  ax = Axis(fig[1, 1], ylabel=L"\rho(r)", title=L"\text{Varying}~\alpha~\text{with}~%$(p2tex(p))")
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
  saveFig(fig, "varying-parameters", p)
  return fig
end

function plotSimulationHistograms(p=Params.defaultParams, runSim=true)
  if runSim
    runSimulator(p, 2000, true)
  end

  fig = Figure()
  df = CSV.read("/tmp/positions.csv", DataFrames.DataFrame, header=false)
  dimension = length(axes(df, 2))
  center = [sum(df[!, k]) / length(df[!, k]) for k in 1:dimension]
  @show center
  radialDistance = hypot.([df[!, k] .- center[k] for k in 1:dimension]...)
  ax = Axis(fig[1, 1], xlabel=L"\text{Radial distance}~r", ylabel=LT"Density",
    title=L"\text{Simulation Output Positional Distribution with}~%$(p2tex(p))")
  hist!(ax, radialDistance, bins=0:0.02:(maximum(radialDistance)*1.05), color=dissertationColours[1])

  # TODO:
  # df = CSV.read("/tmp/position-histogram.csv", DataFrames.DataFrame, header=["hist"])
  # ax = Axis(fig[2, 1], xlabel=L"\text{Radial distance}~r", ylabel=LT"Density",
  #   title=L"\text{Averaged Simulation Output Distribution over 20 runs}")
  # barplot!(ax, df.hist, gap=0, color=dissertationColours[2])

  df = CSV.read("/tmp/velocities.csv", DataFrames.DataFrame, header=false)
  velocity = hypot.([df[!, k] for k in 1:dimension]...)
  ax = Axis(fig[2, 1], xlabel=L"\text{Velocity}~v", ylabel=LT"Density",
    title=L"\text{Simulation Output Velocity Distribution}")
  hist!(ax, velocity, bins=20, color=dissertationColours[2])

  saveFig(fig, "simulation-histogram", p)
  return fig
end

function plotSimulationAndSolverComparison(p::Params.Parameters=Params.known2dParams; runSim=true, big=true)
  env = Utils.createEnvironment(p)
  if runSim
    runSimulator(env.p, 2000, big)
  end

  N = 100
  df = CSV.read("/tmp/positions.csv", DataFrames.DataFrame, header=false)
  dimension = length(axes(df, 2))
  @show center = [sum(df[!, k]) / length(df[!, k]) for k in 1:dimension]
  radialDistance = hypot.([df[!, k] .- center[k] for k in 1:dimension]...)
  pseudoRadialDistance = radialDistance .* sign.(df[!, 1] .- center[1])  # signs according to sign(x)
  @show maxR = maximum(radialDistance)
  @show R_opt = Solver.outerOptimisation(Int64(N / 2), env).minimizer[1]
  solution = Utils.rho(x_vec_noends, Solver.solve(N, R_opt, env), env)
  x = x_vec_noends * maxR

  fig = Figure()
  ax = Axis(fig[1, 1], xlabel=L"\text{Pseudo radial distance}~r \cdot \mathrm{sign}(x_1)", ylabel=LT"Density",
    title=L"\text{Particle Simulation Output Distribution with}~%$(p2tex(p))")
  hist!(ax, pseudoRadialDistance, bins=LinRange([-1, 1] * maxR..., 20), scale_to=maximum(solution),
    label=LT"Particle Simulation", color=dissertationColours[1])
  lines!(ax, x, solution, color=dissertationColours[4], linewidth=3.0, label=LT"Spectral Method")
  ylims!(ax, 0, maximum(solution) * 1.12)
  axislegend(ax, position=:lb)
  saveFig(fig, "simulation-solver-comparison", p)
  return fig
end

function plotSimulationQuiver(p::Params.Parameters=Params.known2dParams; iterations::Int64=4000, runSim=true)
  if runSim
    runSimulator(p, iterations, false)
  end
  posidf, velodf, dimension = loadSimulatorData()
  @assert dimension >= 2

  f = (p.friction.selfPropulsion > 0) ? 20 : 2
  fig = Figure()
  ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y", title=L"\text{Simulation Output with}~%$(p2tex(p))")
  scatter!(ax, posidf[!, 1], posidf[!, 2], color=dissertationColours[1])
  quiver!(ax, posidf[!, 1], posidf[!, 2], velodf[!, 1] / f, velodf[!, 2] / f,
    color=velodf[!, 1], linewidth=2)
  saveFig(fig, "simulation-quiver", p)
  return fig
end

function plot3dSimulationQuiver(p::Params.Parameters=Params.morsePotiSwarming3d; iterations::Int64=12000, runSim=true, fcc=false, withQuiver=true)
  if runSim
    runSimulator(p, iterations, false)
  end
  posidf, velodf, dimension = loadSimulatorData()
  @assert dimension >= 3

  f = 35
  fig = Figure()
  ax = Axis3(fig[1, 1], xlabel=L"x", ylabel=L"y", zlabel=L"z", title=L"\text{Simulation Output with}~%$(p2tex(p))")
  scatter!(ax, posidf[!, 1], posidf[!, 2], posidf[!, 3], color=dissertationColours[1], markersize=fcc ? 50 : (withQuiver ? 10 : 20))
  if fcc
    lines!(ax, posidf[!, 1], posidf[!, 2], posidf[!, 3], color=:black)
  elseif withQuiver
    quiver!(ax, posidf[!, 1], posidf[!, 2], posidf[!, 3], velodf[!, 1] / f, velodf[!, 2] / f, velodf[!, 3] / f,
      color=velodf[!, 1], linewidth=0.007, arrowsize=Vec3f(0.3, 0.3, 0.4) * 0.07, fxaa=true)
  end
  if fcc
    saveFig(fig, "simulation-fcc-3d", p; throughEps=true)
  else
    saveFig(fig, "simulation-quiver-3d", p; throughEps=true)
  end
  return fig
end

function plotPhaseSpace(p=Params.defaultParams; runSim=true)
  env = Utils.createEnvironment(p)
  if runSim
    runSimulator(p)
  end

  fig = Figure()
  posidf, velodf, dimension = loadSimulatorData()
  ax = Axis(fig[1, 1], xlabel=L"\text{First coordinate}~x", ylabel=L"\text{First velocity component}~v_x",
    title=L"\text{Simulation Output Phase Space with}~%$(p2tex(p))")
  scatter!(ax, posidf[!, 1], velodf[!, 1], color=posidf[!, 1])
  ax = Axis(fig[2, 1], xlabel=L"\text{Radial distance}~r", ylabel=L"\text{Velocity}~v",
    title=L"\text{Simulation Output Phase Space with}~%$(p2tex(p))")
  radialDistance = hypot.([posidf[!, k] for k in 1:dimension]...)
  velocity = hypot.([velodf[!, k] for k in 1:dimension]...)
  scatter!(ax, radialDistance, velocity, color=posidf[!, 1])
  saveFig(fig, "phase-space-plot", p)
  return fig
end

function plotConditionNumberGrowth(p=Params.defaultParams)
  Ns = 2 .^ (1:8)
  env = Utils.createEnvironment(p)
  OpCond(N) = Utils.opCond(Solver.constructOperatorFromEnv(N, p.R0, env))
  opconds = OpCond.(Ns)
  fig = Figure(resolution=(800, 500))
  ax = Axis(fig[1, 1], xlabel=L"\text{Matrix size}~N", ylabel=L"\text{Condition number}~\kappa(Q)",
    xscale=log10, yscale=log10, title=L"\text{Growth of the condition number with}~%$(p2tex(p))", xticks=Ns)
  lines!(ax, Ns, opconds)
  scatter!(ax, Ns, opconds, label=LT"Full Operator")
  if isa(p.potential, Params.AttractiveRepulsive)
    AttOpCond(N) = Utils.opCond(AttractiveRepulsiveSolver.recursivelyConstructOperator(N, p.potential.alpha, env))
    opconds = AttOpCond.(Ns)
    lines!(ax, Ns, opconds)
    scatter!(ax, Ns, opconds, label=LT"Attractive Operator")
    RepOpCond(N) = Utils.opCond(AttractiveRepulsiveSolver.recursivelyConstructOperator(N, p.potential.beta, env))
    opconds = RepOpCond.(Ns)
    lines!(ax, Ns, opconds)
    scatter!(ax, Ns, opconds, label=LT"Repulsive Operator")
  end
  axislegend(ax)
  saveFig(fig, "condition-number-growth", p)
  return fig
end

function plotCoefficients(p=Params.morsePotiParams; N=200)
  env = Utils.createEnvironment(p)
  solution = Solver.solve(N, p.R0, env)
  fig = Figure(resolution=(800, 500))
  ax = Axis(fig[1, 1], xlabel=L"\text{Index}~k", ylabel=L"\text{Abs. Coefficient}~|\rho_k|",
    yscale=log10, title=L"\text{Solution coefficients with}~%$(p2tex(p))")
  scatter!(ax, 1:N, abs.(solution), label=LT"without regularisation")
  for s in [1e-8, 1e-6, 1e-4, 1e-2]
    regSolution = Solver.solveWithRegularisation(N, env.p.R0, env, s)
    scatter!(ax, 1:N, abs.(regSolution), label=L"\text{with regularisation}~s=10^{%$(Int64(log10(s)))}")
  end
  axislegend(ax)
  saveFig(fig, "coefficients", p)
  return fig
end

function plotAnalyticErrorVaryingRegularisation(p=Params.knownAnalyticParams)
  # TODO
end

function plotGeneralNGErrorMatrix(p=Params.morsePotiParams)
  maxN = 15
  maxG = 15
  errorMatrix = zeros(maxN, maxG)
  for N in 1:maxN
    for G in 1:maxG
      errorMatrix[N, G] = 1
    end
  end
  # TODO: do the same for runtime

  fig = enhancedSpyPlot(errorMatrix, L"N G \text{Error Matrix}")
  saveFig(fig, "N-G-error-matrix", p)
  return fig
end

function plotMultivariateEnergy(d=1, m=1)
  alpha_vec = 0.5:0.11:(2+2*m-d)
  beta = 0.4
  R_vec = 0.02:0.05:2.2
  E = zeros(length(alpha_vec), length(R_vec))
  for k in eachindex(alpha_vec)
    # p = Params.Parameters(d=d, m=m, potential=Params.AttractiveRepulsive(alpha=2 + 2 * m - d - 0.1, beta=alpha_vec[k]))
    p = Params.Parameters(d=d, m=m, potential=Params.AttractiveRepulsive(alpha=alpha_vec[k], beta=beta))
    Params.checkParameters(p)
    env = Utils.createEnvironment(p)
    F(R) = Utils.totalEnergy(Solver.solve(8, R, env), R, env)
    E[k, :] = F.(R_vec)
  end
  println("Constructed E. Fitting.")

  fitFunction(alpha, R; x) = x[1] + alpha * x[2] + alpha^2 * x[3] + alpha^3 * x[4] + log(alpha) * x[13] +
                             R^alpha * x[5] + R^beta * x[6] + R * x[7] + R^2 * x[8] + R^3 * x[9] + R^4 * x[10] +
                             R^(alpha - 1) * x[11] + R^(beta - 1) * x[12]
  #  SpecialFunctions.beta(0.5, (3 - beta) / 2)^(1 / (beta - 2)) * x[14]
  #  SpecialFunctions.beta(0.5, (3 - alpha) / 2)^(1 / (alpha - 2)) * x[15]
  #  R^(4 - alpha) * SpecialFunctions.beta(0.5, (3 - alpha) / 2)^(1 / (alpha - 2)) * cos(alpha * pi / 2) * x[16]
  squareError(x) = sum((fitFunction.(alpha_vec, R_vec'; x=x) - E) .^ 2)
  result = Optim.optimize(squareError, zeros(14), method=Optim.LBFGS(), iterations=2000; autodiff=:forward)
  @show result
  @show result.minimizer, result.minimum

  fig = Figure(resolution=(800, 600))
  ax = Axis3(fig[1, 1], aspect=:data, perspectiveness=0.2, elevation=π / 9,
    xzpanelcolor=(:black, 0.75), yzpanelcolor=(:black, 0.75), azimuth=0.2 * pi, viewmode=:stretch, protrusions=(50, 10, 30, 0),
    zgridcolor=:grey, ygridcolor=:grey, xgridcolor=:grey, xlabel=L"\alpha", ylabel=L"R", zlabel=L"E(\alpha, R)")
  Emin, Emax = minimum(E), maximum(E)
  surface!(ax, alpha_vec, R_vec, E; colormap=:viridis, colorrange=(Emin, Emax), transparency=true)
  scatter!(ax, alpha_vec, R_vec, fitFunction.(alpha_vec, R_vec'; x=result.minimizer); color=dissertationColours[1])
  saveFig(fig, "multivariate-energy-3d", Params.defaultParams, throughEps=true)
  return fig
end

function plotJacobiConvergence()
  B = Utils.defaultEnv.B
  P = Utils.defaultEnv.P
  f(x) = exp(x^2) # function we want to expand
  f_N = P[:, 1:10] \ f.(axes(P, 1))
  fig = Figure(resolution=(800, 420))
  ax = Axis(fig[1, 1], yscale=log10, xlabel=L"r", ylabel=LT"Absolute Error",
    title=L"\text{Expansion of the function}~f(r) = \exp(r^2)~\text{in the radial}~P_k^{(%$(round(B.a, digits=2)), %$(round(B.b, digits=2)))}~\text{basis}")
  for k in 2:10
    lines!(ax, r_vec, vec(abs.(sum(f_N[1:k] .* P[r_vec, 1:k]', dims=1) - f.(r_vec)')), label=L"N = %$k")
    # lines!(ax, r_vec, vec(sum(f_N[1:k] .* P[r_vec, 1:k]', dims=1) - f.(r_vec)'), label=L"N = %$k")
  end
  axislegend(ax)
  saveFig(fig, "jacobi-expansions")
  return fig
end

function plotAll()
  global _extra_pdf
  _extra_pdf = false
  try
    plotJacobiConvergence()
    plotDifferentOrderSolutions(Params.defaultParams)
    plotDifferentOrderSolutions(Params.morsePotiParams)
    plotDifferentOrderSolutions(Params.bumpParams)
    plotAttRepOperators(Params.defaultParams)  # or maybe 2d?
    plotFullOperator(Params.defaultParams)
    plotFullOperator(Params.morsePotiParams)
    plotOuterOptimisation(Params.knownAnalyticParams)
    plotAnalyticSolution(Params.knownAnalyticParams)
    plotStepByStepConvergence(Params.defaultParams)
    plotConvergenceToAnalytic(Params.knownAnalyticParams)
    plotDefaultParameterVariations()  # uses defaultParams hence the name
    plotPhaseSpace(Params.morsePotiParams)
    plotSimulationHistograms(Params.defaultParams)
    plotSimulationQuiver(Params.known2dParams)
    plotSimulationQuiver(Params.morsePotiSwarming2d; iterations=8000)
    plotSpatialEnergyDependence(Params.defaultParams)
    plotVaryingRSolutions(Params.morsePotiParams)
    plotGeneralSolutionApproximation(Params.morsePotiParams)
    plot3dSimulationQuiver(Params.morsePotiSwarming3d)
    plot3dSimulationQuiver(Params.gyroscope3dParams; withQuiver=false)
    plotMonomialBasisConvergence(Params.morsePotiParams)
    plotSimulationAndSolverComparison(Params.morsePotiParams)
    plotConditionNumberGrowth(Params.defaultParams)
    plotCoefficients(Params.morsePotiParams)
  finally
    _extra_pdf = true
  end
  return
end

function plotAllForP(p::Params.Parameters, quick=true)
  plotFullOperator(p)
  plotDifferentOrderSolutions(p)
  plotOuterOptimisation(p)
  plotStepByStepConvergence(p)
  plotVaryingRSolutions(p)
  plotSimulationAndSolverComparison(p; big=false)
  if p.d >= 2
    plotSimulationQuiver(p; runSim=false)
  end
  plotPhaseSpace(p; runSim=false)
  if !quick
    plotSimulationHistograms(p)
    plotMonomialBasisConvergence(p)
    plotGeneralSolutionApproximation(p)
    plotSpatialEnergyDependence(p)
    plotConditionNumberGrowth(p)
    plotCoefficients(p)
  end
  return
end

println("All plotted for today?")
