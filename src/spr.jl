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
    SparsePoleRepresentation <: AbstractBasis    

Sparse pole representation (SPR).
The poles are the extrema of V'_{L-1}(ω).
"""
struct SparsePoleRepresentation{S<:Statistics,B<:AbstractBasis{S},T<:AbstractFloat,FMAT<:SVD
                                } <: AbstractBasis{S}
    basis  :: B
    poles  :: Vector{T}
    u      :: TauPoleBasis{S}
    uhat   :: MatsubaraPoleBasis{S}
    fitmat :: Matrix{Float64}
    matrix :: FMAT
end

function SparsePoleRepresentation(b::AbstractBasis, poles=default_omega_sampling_points(b))
    u = TauPoleBasis(statistics(b), β(b), poles)
    uhat = MatsubaraPoleBasis(statistics(b), β(b), poles)

    # Fitting matrix from IR
    fitmat = -b.s .* b.v(poles)

    # Now, here we *know* that fitmat is ill-conditioned in very particular way:
    # it is a product A * B * C, where B is well conditioned and A, C are scalings.
    return SparsePoleRepresentation(b, poles, u, uhat, fitmat, svd(fitmat; alg = QRIteration()))
end

Base.show(io::IO, spr::SparsePoleRepresentation) =
    print(io, "SparsePoleRepresentation for $(spr.basis) with poles at $(spr.poles)")

Base.length(spr::SparsePoleRepresentation) = length(spr.poles)
Base.size(spr::SparsePoleRepresentation) = (length(spr), )

β(spr::SparsePoleRepresentation) = β(spr.basis)
ωmax(spr::SparsePoleRepresentation) = ωmax(spr.basis)
Λ(spr::SparsePoleRepresentation) = Λ(spr.basis)

sampling_points(spr::SparsePoleRepresentation) = spr.poles
significance(spr::SparsePoleRepresentation) = ones(size(spr))
accuracy(spr::SparsePoleRepresentation) = accuracy(spr.basis)

default_tau_sampling_points(spr::SparsePoleRepresentation) = 
    default_tau_sampling_points(spr.basis)
default_matsubara_sampling_points(spr::SparsePoleRepresentation) =
    default_matsubara_sampling_points(spr.basis)
iswellconditioned(::SparsePoleRepresentation) = false

"""
    from_IR(spr::SparsePoleRepresentation, gl::AbstractArray, dims=1)

From IR to SPR. `gl``: Expansion coefficients in IR.
"""
function from_IR(spr::SparsePoleRepresentation, gl::AbstractArray, dims=1)
    return mapslices(sl -> spr.matrix \ sl, gl; dims)
end

"""
    to_IR(spr::SparsePoleRepresentation, g_spr::AbstractArray, dims=1)

From SPR to IR. `g_spr``: Expansion coefficients in SPR.
"""
function to_IR(spr::SparsePoleRepresentation, g_spr::AbstractArray, dims=1)
    return mapslices(sl -> spr.fitmat * sl, g_spr; dims)
end
