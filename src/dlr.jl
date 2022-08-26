struct MatsubaraPoleBasis{S<:Statistics} <: AbstractBasis{S}
    β     :: Float64
    poles :: Vector{Float64}
    MatsubaraPoleBasis(statistics::Statistics, β::Real, poles::Vector{<:Real}) =
        new{typeof(statistics)}(β, poles)
end

(b::MatsubaraPoleBasis{Fermionic})(n::FermionicFreq) = 
    @. 1 / (valueim(n, β(b)) - b.poles)
(b::MatsubaraPoleBasis{Bosonic})(n::BosonicFreq) = 
    @. tanh(β(b) / 2 * b.poles) / (valueim(n, β(b)) - b.poles)

(b::MatsubaraPoleBasis{S})(n::AbstractVector{MatsubaraFreq{S}}) where {S} =
    mapreduce(b, hcat, n)
(b::MatsubaraPoleBasis)(n::AbstractVector{<:Integer}) =
    b(MatsubaraFreq.(n))


struct TauPoleBasis{S<:Statistics} <: AbstractBasis{S}
    β          :: Float64
    poles      :: Vector{Float64}
    ωmax       :: Float64
    TauPoleBasis(statistics::Statistics, β::Real, poles::Vector{<:Real}) =
        new{typeof(statistics)}(β, poles, maximum(abs, poles))
end

ωmax(basis::TauPoleBasis) = basis.ωmax

function (basis::TauPoleBasis)(τ::Vector{<:Real})
    all(τ -> 0 ≤ τ ≤ β(basis), τ) || throw(DomainError(τ, "τ must be in [0, β]."))

    x = 2τ ./ β(basis) .- 1
    y = basis.poles ./ ωmax(basis)
    Λ = β(basis) * ωmax(basis)
    
    res = -LogisticKernel(Λ).(x, transpose(y))
    return transpose(res)
end

"""
    DiscreteLehmannRepresentation <: AbstractBasis

Discrete Lehmann representation (DLR) with poles selected according to extrema of IR.

This class implements a variant of the discrete Lehmann representation (`DLR`) [1].  Instead
of a truncated singular value expansion of the analytic continuation kernel ``K`` like the IR,
the discrete Lehmann representation is based on a "sketching" of ``K``.  The resulting basis
is a linear combination of discrete set of poles on the real-frequency axis, continued to the
imaginary-frequency axis:

     G(iv) == sum(a[i] / (iv - w[i]) for i in range(L))

Warning
    The poles on the real-frequency axis selected for the DLR are based on a rank-revealing
    decomposition, which offers accuracy guarantees.  Here, we instead select the pole locations
    based on the zeros of the IR basis functions on the real axis, which is a heuristic.  We do not
    expect that difference to matter, but please don't blame the DLR authors if we were wrong :-)

[1]: https://doi.org/10.48550/arXiv.2110.06765
"""
struct DiscreteLehmannRepresentation{S<:Statistics,B<:AbstractBasis{S},T<:AbstractFloat,FMAT<:SVD
                                } <: AbstractBasis{S}
    basis  :: B
    poles  :: Vector{T}
    u      :: TauPoleBasis{S}
    uhat   :: MatsubaraPoleBasis{S}
    fitmat :: Matrix{Float64}
    matrix :: FMAT
end

function DiscreteLehmannRepresentation(b::AbstractBasis, poles=default_omega_sampling_points(b))
    u = TauPoleBasis(statistics(b), β(b), poles)
    uhat = MatsubaraPoleBasis(statistics(b), β(b), poles)

    # Fitting matrix from IR
    fitmat = -b.s .* b.v(poles)

    # Now, here we *know* that fitmat is ill-conditioned in very particular way:
    # it is a product A * B * C, where B is well conditioned and A, C are scalings.
    return DiscreteLehmannRepresentation(b, poles, u, uhat, fitmat, svd(fitmat; alg = QRIteration()))
end

Base.show(io::IO, dlr::DiscreteLehmannRepresentation) =
    print(io, "DiscreteLehmannRepresentation for $(dlr.basis) with poles at $(dlr.poles)")

Base.length(dlr::DiscreteLehmannRepresentation) = length(dlr.poles)
Base.size(dlr::DiscreteLehmannRepresentation) = (length(dlr), )

β(dlr::DiscreteLehmannRepresentation) = β(dlr.basis)
ωmax(dlr::DiscreteLehmannRepresentation) = ωmax(dlr.basis)
Λ(dlr::DiscreteLehmannRepresentation) = Λ(dlr.basis)

sampling_points(dlr::DiscreteLehmannRepresentation) = dlr.poles
significance(dlr::DiscreteLehmannRepresentation) = ones(size(dlr))
accuracy(dlr::DiscreteLehmannRepresentation) = accuracy(dlr.basis)

default_tau_sampling_points(dlr::DiscreteLehmannRepresentation) = 
    default_tau_sampling_points(dlr.basis)
default_matsubara_sampling_points(dlr::DiscreteLehmannRepresentation) =
    default_matsubara_sampling_points(dlr.basis)
iswellconditioned(::DiscreteLehmannRepresentation) = false

"""
    from_IR(dlr::DiscreteLehmannRepresentation, gl::AbstractArray, dims=1)

From IR to DLR. `gl``: Expansion coefficients in IR.
"""
function from_IR(dlr::DiscreteLehmannRepresentation, gl::AbstractArray, dims=1)
    return mapslices(sl -> dlr.matrix \ sl, gl; dims)
end

"""
    to_IR(dlr::DiscreteLehmannRepresentation, g_dlr::AbstractArray, dims=1)

From DLR to IR. `g_dlr``: Expansion coefficients in DLR.
"""
function to_IR(dlr::DiscreteLehmannRepresentation, g_dlr::AbstractArray, dims=1)
    return mapslices(sl -> dlr.fitmat * sl, g_dlr; dims)
end
