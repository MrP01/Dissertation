using ClassicalOrthogonalPolynomials, ContinuumArrays, Plots, Formatting, HypergeometricFunctions, SpecialFunctions
import ContinuumArrays: MappedWeightedBasisLayout, Map, WeightedBasisLayout

d = 1;  # dimension
m = 2  # integer
alpha = 2.0
beta = 8.0
@assert -d < alpha < 2 + 2 * m - d
@assert beta > -d
@assert m >= 0 && m == floor(m)

function theorem216(r, n)
  @show r, n
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

# represent the basis P_n^(a,b)(2r^2-1)
P = Jacobi(m - (alpha + d) / 2, (d - 2) / 2)[QuadraticMap(), :]
r = axes(P, 1)

N = 4  # order of basis expansion
BigMatrix = zeros(N, N);
for column in 1:N
  func = theorem216(r, column)
  f_N = P[:, 1:N] \ func
  BigMatrix[:, column] .= f_N
end

BigRHS = zeros(N);
BigRHS[1] = 1;

@show BigMatrix
BigSolution = BigMatrix \ BigRHS;

x = 0:0.01:1
y = vec(sum(BigSolution .* P[x, 1:N]', dims=1));
plot(x, y)
