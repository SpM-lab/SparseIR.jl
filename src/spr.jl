struct MatsubaraPoleBasis{S<:Statistics} <: AbstractBasis
    β          :: Float64
    statistics :: S
    poles      :: Vector{Float64}
end

β(basis::MatsubaraPoleBasis) = basis.β
statistics(basis::MatsubaraPoleBasis) = basis.statistics

function (basis::MatsubaraPoleBasis{S})(n::MatsubaraFreq{S}) where {S}
    beta = β(basis)
    iν = valueim(n, beta)
    if S === Fermionic
        return @. 1 / (iν - basis.poles)
    else
        return @. tanh(0.5beta * basis.poles) / (iν - basis.poles)
    end
end
(basis::MatsubaraPoleBasis{S})(n::AbstractVector{MatsubaraFreq{S}}) where {S} = 
    mapreduce(basis, hcat, n)
(basis::MatsubaraPoleBasis)(n::AbstractVector{<:Integer}) = basis(MatsubaraFreq.(n))

struct TauPoleBasis{S<:Statistics} <: AbstractBasis
    β          :: Float64
    poles      :: Vector{Float64}
    statistics :: S
    ωmax       :: Float64
end

ωmax(basis::TauPoleBasis) = basis.ωmax

function TauPoleBasis(beta::Real, statistics::Statistics, poles::Vector{<:AbstractFloat})
    return TauPoleBasis(beta, poles, statistics, maximum(abs, poles))
end

function (basis::TauPoleBasis)(τ::Vector{<:AbstractFloat})
    all(τ -> 0 ≤ τ ≤ beta(basis), τ) || throw(DomainError(τ, "τ must be in [0, beta]!"))

    x = (2 / beta(basis)) .* τ .- 1
    y = basis.poles ./ ωmax(basis)
    Λ = beta(basis) * ωmax(basis)
    return -transpose(LogisticKernel(Λ).(x, transpose(y)))
end

"""
    SparsePoleRepresentation <: AbstractBasis    

Sparse pole representation.
"""
struct SparsePoleRepresentation{B<:AbstractBasis,T<:AbstractFloat,S<:Statistics,FMAT<:SVD} <: AbstractBasis
    basis      :: B
    poles      :: Vector{T}
    u          :: TauPoleBasis{S}
    uhat       :: MatsubaraPoleBasis{S}
    statistics :: S
    fitmat     :: Matrix{Float64}
    matrix     :: FMAT
end

function Base.show(io::IO, obj::SparsePoleRepresentation)
    return print(io, "SparsePoleRepresentation for $(obj.basis) with poles at $(obj.poles)")
end

function SparsePoleRepresentation(basis::AbstractBasis,
                                  poles=default_omega_sampling_points(basis))
    u = TauPoleBasis(β(basis), statistics(basis), poles)
    uhat = MatsubaraPoleBasis(β(basis), statistics(basis), poles)
    fitmat = -basis.s .* basis.v(poles)
    return SparsePoleRepresentation(basis, poles, u, uhat, 
                                    statistics(basis), fitmat, svd(fitmat))
end

β(obj::SparsePoleRepresentation) = β(obj.basis)

function Base.getproperty(obj::SparsePoleRepresentation, d::Symbol)
    if d === :v
        return nothing
    else
        return getfield(obj, d)
    end
end

default_tau_sampling_points(obj::SparsePoleRepresentation) = 
    default_tau_sampling_points(obj.basis)

default_matsubara_sampling_points(obj::SparsePoleRepresentation) = 
    default_matsubara_sampling_points(obj.basis)

iswellconditioned(::SparsePoleRepresentation) = false

"""
    from_IR(spr::SparsePoleRepresentation, gl::AbstractArray, dims=1)

From IR to SPR.
"""
function from_IR(spr::SparsePoleRepresentation, gl::AbstractArray, dims=1)
    return mapslices(sl -> spr.matrix \ sl, gl; dims)
end

"""
    to_IR(spr::SparsePoleRepresentation, g_spr::AbstractArray, dims=1)

From SPR to IR.
"""
function to_IR(spr::SparsePoleRepresentation, g_spr::AbstractArray, dims=1)
    return mapslices(sl -> spr.fitmat * sl, g_spr; dims)
end
