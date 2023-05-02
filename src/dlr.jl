struct MatsubaraPoles{S<:Statistics}
    β::Float64
    poles::Vector{Float64}
    function MatsubaraPoles{S}(β::Real, poles::Vector{<:Real}) where {S<:Statistics}
        new{S}(β, poles)
    end
end

function MatsubaraPoles(stats::Statistics, β::Real, poles::Vector{<:Real})
    MatsubaraPoles{typeof(stats)}(β, poles)
end

(mp::MatsubaraPoles{Fermionic})(n::FermionicFreq) = 1 ./ (valueim(n, mp.β) .- mp.poles)
function (mp::MatsubaraPoles{Bosonic})(n::BosonicFreq)
    tanh.(mp.β / 2 .* mp.poles) ./ (valueim(n, mp.β) .- mp.poles)
end

function (mp::MatsubaraPoles{S})(n::AbstractVector{MatsubaraFreq{S}}) where {S}
    mapreduce(mp, hcat, n)
end
(mp::MatsubaraPoles)(n::AbstractVector{<:Integer}) = mp(MatsubaraFreq.(n))

struct TauPoles{S<:Statistics}
    β::Float64
    poles::Vector{Float64}
    ωmax::Float64
    function TauPoles{S}(β::Real, poles::Vector{<:Real}) where {S<:Statistics}
        new{S}(β, poles, maximum(abs, poles))
    end
end

function TauPoles(stats::Statistics, β::Real, poles::Vector{<:Real})
    TauPoles{typeof(stats)}(β, poles)
end

function (tp::TauPoles)(τ::Vector{<:Real})
    all(τ -> 0 ≤ τ ≤ tp.β, τ) || throw(DomainError(τ, "τ must be in [0, β]."))

    x = reshape(2τ ./ tp.β .- 1, (1, :))
    y = tp.poles ./ tp.ωmax
    Λ = tp.β * tp.ωmax

    .-LogisticKernel(Λ).(x, y)
end

"""
    DiscreteLehmannRepresentation <: AbstractBasis

Discrete Lehmann representation (DLR) with poles selected according to extrema of IR.

This class implements a variant of the discrete Lehmann representation (`DLR`) [1](https://doi.org/10.48550/arXiv.2110.06765). Instead
of a truncated singular value expansion of the analytic continuation kernel ``K`` like the IR,
the discrete Lehmann representation is based on a "sketching" of ``K``. The resulting basis
is a linear combination of discrete set of poles on the real-frequency axis, continued to the
imaginary-frequency axis:

     G(iv) == sum(a[i] / (iv - w[i]) for i in range(L))

Warning
The poles on the real-frequency axis selected for the DLR are based on a rank-revealing
decomposition, which offers accuracy guarantees. Here, we instead select the pole locations
based on the zeros of the IR basis functions on the real axis, which is a heuristic. We do not
expect that difference to matter, but please don't blame the DLR authors if we were wrong :-)
"""
struct DiscreteLehmannRepresentation{S<:Statistics,B<:AbstractBasis{S},T<:AbstractFloat,
                                     FMAT<:SVD} <: AbstractBasis{S}
    basis  :: B
    poles  :: Vector{T}
    u      :: TauPoles{S}
    uhat   :: MatsubaraPoles{S}
    fitmat :: Matrix{Float64}
    matrix :: FMAT
end

function DiscreteLehmannRepresentation(b::AbstractBasis,
                                       poles=default_omega_sampling_points(b))
    u = TauPoles(statistics(b), β(b), poles)
    uhat = MatsubaraPoles(statistics(b), β(b), poles)

    # Fitting matrix from IR
    fitmat = -b.s .* b.v(poles)

    # Now, here we *know* that fitmat is ill-conditioned in very particular way:
    # it is a product A * B * C, where B is well conditioned and A, C are scalings.
    DiscreteLehmannRepresentation(b, poles, u, uhat, fitmat, svd(fitmat; alg=QRIteration()))
end

function Base.show(io::IO, ::MIME"text/plain", dlr::DiscreteLehmannRepresentation)
    print(io, "DiscreteLehmannRepresentation for $(dlr.basis) with poles at $(dlr.poles)")
end

Base.length(dlr::DiscreteLehmannRepresentation) = length(dlr.poles)
Base.size(dlr::DiscreteLehmannRepresentation) = (length(dlr),)

β(dlr::DiscreteLehmannRepresentation) = β(dlr.basis)
ωmax(dlr::DiscreteLehmannRepresentation) = ωmax(dlr.basis)
Λ(dlr::DiscreteLehmannRepresentation) = Λ(dlr.basis)

sampling_points(dlr::DiscreteLehmannRepresentation) = dlr.poles
significance(dlr::DiscreteLehmannRepresentation) = ones(size(dlr))
accuracy(dlr::DiscreteLehmannRepresentation) = accuracy(dlr.basis)

function default_tau_sampling_points(dlr::DiscreteLehmannRepresentation)
    default_tau_sampling_points(dlr.basis)
end
function default_matsubara_sampling_points(dlr::DiscreteLehmannRepresentation; kwargs...)
    default_matsubara_sampling_points(dlr.basis; kwargs...)
end
iswellconditioned(::DiscreteLehmannRepresentation) = false

"""
    from_IR(dlr::DiscreteLehmannRepresentation, gl::AbstractArray, dims=1)

From IR to DLR. `gl``: Expansion coefficients in IR.
"""
function from_IR(dlr::DiscreteLehmannRepresentation, gl::AbstractArray, dims=1)
    mapslices(sl -> dlr.matrix \ sl, gl; dims)
end

"""
    to_IR(dlr::DiscreteLehmannRepresentation, g_dlr::AbstractArray, dims=1)

From DLR to IR. `g_dlr``: Expansion coefficients in DLR.
"""
function to_IR(dlr::DiscreteLehmannRepresentation, g_dlr::AbstractArray, dims=1)
    mapslices(sl -> dlr.fitmat * sl, g_dlr; dims)
end
