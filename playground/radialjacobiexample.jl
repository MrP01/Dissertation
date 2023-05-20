using ClassicalOrthogonalPolynomials, ContinuumArrays
import ContinuumArrays: MappedWeightedBasisLayout, Map, WeightedBasisLayout

# These definitions allow the use of the radially shifted Jacobi bases
struct QuadraticMap{T} <: Map{T} end
struct InvQuadraticMap{T} <: Map{T} end
QuadraticMap() = QuadraticMap{Float64}()
InvQuadraticMap() = InvQuadraticMap{Float64}()
Base.getindex(::QuadraticMap, r::Number) = 2r^2-1
Base.axes(::QuadraticMap{T}) where T = (Inclusion(0..1),)
Base.axes(::InvQuadraticMap{T}) where T = (Inclusion(-1..1),)
Base.getindex(d::InvQuadraticMap, x::Number) = sqrt((x+1)/2)
ContinuumArrays.invmap(::QuadraticMap{T}) where T = InvQuadraticMap{T}()
ContinuumArrays.invmap(::InvQuadraticMap{T}) where T = QuadraticMap{T}()
Base.getindex(d::QuadraticMap, x::Inclusion) = d
Base.getindex(d::InvQuadraticMap, x::Inclusion) = d

# set a and b
a = 1.4
b = 3.2
# represent the basis P_n^(a,b)(2r^2-1)
P =  Jacobi(a,b)[QuadraticMap(),:]
r = axes(P,1)

# the following are examples on how to compute the expansion coefficients of a function in this basis
# note: the first element is the 0-th degree coefficient, the second element is the 1st degree coefficient etc.
f(x) = exp(x^2) # function we want to expand
f_n = P \ f.(r) # infinite vector which is adaptively computed (the more elements you ask for, the more it computes)

# explicit length of coefficients instead of infinite vector
# such that f(x) ≈ \sum_{k=0}^N f_N[k] * P[:,k][x].
N = 20
f_N = P[:,1:N] \ f.(r)

# Note: Since Julia is just in time (JIT) compiled, some stuff may take a bit on first run but will be very fast on all subsequent executions.