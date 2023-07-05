using HypergeometricFunctions
using SpecialFunctions

d = 1
m = 2
n = 1
alpha = -0.5
beta = 1
@assert -d < alpha < 2 + 2 * m - d
@assert -d < beta
x = 3 + 0im

prefactor =
    pi^(d / 2) *
    gamma(1 + beta / 2) *
    gamma((beta + d) / 2) *
    gamma(m + n - (alpha + d) / 2 + 1) / (
        gamma(d / 2) *
        gamma(n + 1) *
        gamma(beta / 2 - n + 1) *
        gamma((beta - alpha) / 2 + m + n + 1)
    );
integral_value =
    prefactor * HypergeometricFunctions._₂F₁(
        n - beta / 2,
        -m - n + (alpha - beta) / 2,
        d / 2,
        abs(x)^2 + 0im,
    );
println("Integral value: $(integral_value)");
