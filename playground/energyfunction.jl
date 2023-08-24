import GLMakie, Optim, SpecialFunctions

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
    F(R) = Utils.totalEnergy(Solver.solve(8, R, env), R, 0.0, env)
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

  fig = GLMakie.Figure(resolution=(1200, 800))
  ax = GLMakie.Axis3(fig[1, 1], aspect=:data, perspectiveness=0.5, elevation=π / 9,
    xzpanelcolor=(:black, 0.45), yzpanelcolor=(:black, 0.45),
    zgridcolor=:grey, ygridcolor=:grey, xgridcolor=:grey, xlabel=L"\alpha", ylabel=L"R", zlabel=L"E(\alpha, R)")
  Emin, Emax = minimum(E), maximum(E)
  GLMakie.surface!(ax, alpha_vec, R_vec, E; colormap=:viridis, colorrange=(Emin, Emax), transparency=true)
  GLMakie.scatter!(ax, alpha_vec, R_vec, fitFunction.(alpha_vec, R_vec'; x=result.minimizer); colormap=:viridis, colorrange=(Emin, Emax))
  return fig
end
