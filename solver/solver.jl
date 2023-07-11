using ClassicalOrthogonalPolynomials, ContinuumArrays, Plots, Formatting, HypergeometricFunctions, SpecialFunctions
import ContinuumArrays: MappedWeightedBasisLayout, Map, WeightedBasisLayout
gaston()

d = 1  # dimension
m = 2  # integer
alpha = 2.2
beta = 1.6
R = 8  # radius of the interval [-R, R]
@assert -d < alpha < 2 + 2 * m - d
@assert beta > -d
@assert m >= 0 && m == floor(m)

function theorem216(r, n, beta)
  @show r, n, beta
  prefactor =
    pi^(d / 2) *
    gamma(1 + beta / 2) *
    gamma((beta + d) / 2) *
    gamma(m + n - (alpha + d) / 2 + 1) / (
      gamma(d / 2) *
      gamma(n + 1) *
      gamma(beta / 2 - n + 1) *
      gamma((beta - alpha) / 2 + m + n + 1)
    )
  integral_value =
    prefactor * HypergeometricFunctions._₂F₁.(
      n - beta / 2,
      -m - n + (alpha - beta) / 2,
      d / 2,
      r .^ 2,
    )
  @show integral_value
  return integral_value
end

# These definitions allow the use of the radially shifted Jacobi bases
struct QuadraticMap{T} <: Map{T} end
struct InvQuadraticMap{T} <: Map{T} end
QuadraticMap() = QuadraticMap{Float64}()
InvQuadraticMap() = InvQuadraticMap{Float64}()
Base.getindex(::QuadraticMap, r::Number) = 2r^2 - 1
Base.axes(::QuadraticMap{T}) where {T} = (Inclusion(0 .. 1),)
Base.axes(::InvQuadraticMap{T}) where {T} = (Inclusion(-1 .. 1),)
Base.getindex(d::InvQuadraticMap, x::Number) = sqrt((x + 1) / 2)
ContinuumArrays.invmap(::QuadraticMap{T}) where {T} = InvQuadraticMap{T}()
ContinuumArrays.invmap(::InvQuadraticMap{T}) where {T} = QuadraticMap{T}()
Base.getindex(d::QuadraticMap, x::Inclusion) = d
Base.getindex(d::InvQuadraticMap, x::Inclusion) = d

map = QuadraticMap();
imap = InvQuadraticMap();

# represent the basis P_n^(a,b)(2r^2-1)
B = Jacobi(m - (alpha + d) / 2, (d - 2) / 2);
P = B[QuadraticMap(), :];
x = axes(B, 1)
r = axes(P, 1)

@assert x.domain == (-1 .. 1)  # Chebyshev
@assert r.domain == (0 .. 1)  # Radial

function solve(N)
  AttractiveMatrix = zeros(N, N)
  RepulsiveMatrix = zeros(N, N)
  for column in 1:N
    attractiveFunction = theorem216(r, column, alpha)
    repulsiveFunction = theorem216(r, column, beta)
    attractiveExpansionCoeffs = P[:, 1:N] \ attractiveFunction
    repulsiveExpansionCoeffs = P[:, 1:N] \ repulsiveFunction
    AttractiveMatrix[:, column] .= attractiveExpansionCoeffs
    RepulsiveMatrix[:, column] .= repulsiveExpansionCoeffs
  end

  BigMatrix = (R^(alpha + d) / alpha) * AttractiveMatrix - (R^(beta + d) / beta) * RepulsiveMatrix
  BigRHS = zeros(N)
  BigRHS[1] = 1

  BigSolution = BigMatrix \ BigRHS
  return BigSolution
end

r_vec = 0:0.01:1
x_vec = -1:0.01:1

# y_vec_radial = vec(sum(BigSolution .* P[r_vec, 1:N]', dims=1));
plot(x_vec, vec(sum(solve(2) .* P[abs.(x_vec), 1:2]', dims=1)), label="N = 2");
plot!(x_vec, vec(sum(solve(3) .* P[abs.(x_vec), 1:3]', dims=1)), label="N = 3");
plot!(x_vec, vec(sum(solve(4) .* P[abs.(x_vec), 1:4]', dims=1)), label="N = 4");
plot!(x_vec, vec(sum(solve(5) .* P[abs.(x_vec), 1:5]', dims=1)), label="N = 5");

# spy(BigMatrix, ms=20, markershape=:rect)
