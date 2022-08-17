struct MatsubaraPoleBasis{S<:Statistics} <: AbstractBasis{S}
    β          :: Float64
    statistics :: S
    poles      :: Vector{Float64}
end

function (basis::MatsubaraPoleBasis{S})(n::MatsubaraFreq{S}) where {S}
    iν = valueim(n, β(basis))
    if S === Fermionic
        return @. 1 / (iν - basis.poles)
    else
        return @. tanh(β(basis) / 2 * basis.poles) / (iν - basis.poles)
    end
end
function (basis::MatsubaraPoleBasis{S})(n::AbstractVector{MatsubaraFreq{S}}) where {S}
    mapreduce(basis, hcat, n)
end
(basis::MatsubaraPoleBasis)(n::AbstractVector{<:Integer}) = basis(MatsubaraFreq.(n))

struct TauPoleBasis{S<:Statistics} <: AbstractBasis{S}
    β          :: Float64
    poles      :: Vector{Float64}
    statistics :: S
    ωmax       :: Float64
end

ωmax(basis::TauPoleBasis) = basis.ωmax

function TauPoleBasis(β::Real, statistics::Statistics, poles::Vector{<:AbstractFloat})
    return TauPoleBasis(β, poles, statistics, maximum(abs, poles))
end

function (basis::TauPoleBasis)(τ::Vector{<:AbstractFloat})
    all(τ -> 0 ≤ τ ≤ β(basis), τ) || throw(DomainError(τ, "τ must be in [0, β]!"))

    x = (2 / β(basis)) .* τ .- 1
    y = basis.poles ./ ωmax(basis)
    Λ = β(basis) * ωmax(basis)
    return -transpose(LogisticKernel(Λ).(x, transpose(y)))
end

"""
    SparsePoleRepresentation <: AbstractBasis    

Sparse pole representation.
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

# TODO
function Base.show(io::IO, obj::SparsePoleRepresentation)
    return print(io, "SparsePoleRepresentation for $(obj.basis) with poles at $(obj.poles)")
end

function SparsePoleRepresentation(b::AbstractBasis, poles=default_omega_sampling_points(b))
    u = TauPoleBasis(β(b), statistics(b), poles)
    uhat = MatsubaraPoleBasis(β(b), statistics(b), poles)
    fitmat = -b.s .* b.v(poles)
    return SparsePoleRepresentation(b, poles, u, uhat, fitmat, svd(fitmat))
end

β(obj::SparsePoleRepresentation) = β(obj.basis)

function Base.getproperty(obj::SparsePoleRepresentation, d::Symbol)
    if d === :v
        return nothing
    else
        return getfield(obj, d)
    end
end

function default_tau_sampling_points(obj::SparsePoleRepresentation)
    default_tau_sampling_points(obj.basis)
end
function default_matsubara_sampling_points(obj::SparsePoleRepresentation)
    default_matsubara_sampling_points(obj.basis)
end
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
