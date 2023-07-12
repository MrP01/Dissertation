using Plots
include("./solver.jl");
gaston()

const RESULTS_FOLDER = joinpath(@__DIR__, "..", "figures", "results")

r_vec = 0:0.01:1
x_vec = -1:0.01:1

# y_vec_radial = vec(sum(BigSolution .* P[r_vec, 1:N]', dims=1));
plot(x_vec, vec(sum(solve(2) .* P[abs.(x_vec), 1:2]', dims=1)), label="N = 2");
plot!(x_vec, vec(sum(solve(3) .* P[abs.(x_vec), 1:3]', dims=1)), label="N = 3");
plot!(x_vec, vec(sum(solve(4) .* P[abs.(x_vec), 1:4]', dims=1)), label="N = 4");
plot!(x_vec, vec(sum(solve(5) .* P[abs.(x_vec), 1:5]', dims=1)), label="N = 5");
savefig(joinpath(RESULTS_FOLDER, "solution-increasing-order.png"));

spy(constructOperator(20, alpha), ms=20, markershape=:rect)
savefig(joinpath(RESULTS_FOLDER, "attractive-operator.png"));
spy(constructOperator(20, beta), ms=20, markershape=:rect)
savefig(joinpath(RESULTS_FOLDER, "repulsive-operator.png"));

solution = solve(5)
TE(r) = totalEnergy(solution, r)
plot(r_vec[1:end-1], TE.(r_vec[1:end-1]))
savefig(joinpath(RESULTS_FOLDER, "energy-dependence-on-r.png"));
