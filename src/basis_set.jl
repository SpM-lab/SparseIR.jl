"""
    FiniteTempBasisSet

Class for holding IR bases and sparse-sampling objects.

An object of this class holds IR bases for fermions and bosons
and associated sparse-sampling objects.

# Fields
- basis_f::FiniteTempBasis: Fermion basis
- basis_b::FiniteTempBasis: Boson basis
- beta::Float64: Inverse temperature
- wmax::Float64: Cut-off frequency
- tau::Vector{Float64}: Sampling points in the imaginary-time domain
- wn_f::Vector{Int}: Sampling fermionic frequencies
- wn_b::Vector{Int}: Sampling bosonic frequencies
- smpl_tau_f::TauSampling: Sparse sampling for tau & fermion
- smpl_tau_b::TauSampling: Sparse sampling for tau & boson
- smpl_wn_f::MatsubaraSampling: Sparse sampling for Matsubara frequency & fermion
- smpl_wn_b::MatsubaraSampling: Sparse sampling for Matsubara frequency & boson
- sve_result::Tuple{PiecewiseLegendrePoly,Vector{Float64},PiecewiseLegendrePoly}: Results of SVE
"""
struct FiniteTempBasisSet
    basis_f::_DEFAULT_FINITE_TEMP_BASIS
    basis_b::_DEFAULT_FINITE_TEMP_BASIS
    smpl_tau_f::TauSampling{Float64,_DEFAULT_FINITE_TEMP_BASIS,Float64,Float64}
    smpl_tau_b::TauSampling{Float64,_DEFAULT_FINITE_TEMP_BASIS,Float64,Float64}
    smpl_wn_f::MatsubaraSampling{Int,_DEFAULT_FINITE_TEMP_BASIS,ComplexF64,Float64}
    smpl_wn_b::MatsubaraSampling{Int,_DEFAULT_FINITE_TEMP_BASIS,ComplexF64,Float64}
end

"""
    FiniteTempBasisSet(beta, wmax, eps; sve_result=nothing)

Create basis sets for fermion and boson and
associated sampling objects.
Fermion and bosonic bases are constructed by SVE of the logistic kernel.
"""
function FiniteTempBasisSet(beta, wmax, eps; sve_result=nothing)
    if isnothing(sve_result)
        # Create bases by sve of the logistic kernel
        basis_f, basis_b = finite_temp_bases(beta, wmax, eps)
    else
        # Create bases using the given sve results
        basis_f = FiniteTempBasis(fermion, beta, wmax, eps; sve_result)
        basis_b = FiniteTempBasis(boson, beta, wmax, eps; sve_result)
    end

    return FiniteTempBasisSet(
        basis_f,
        basis_b,
        TauSampling(basis_f),
        TauSampling(basis_b),
        MatsubaraSampling(basis_f),
        MatsubaraSampling(basis_b),
    )
end

beta(bset::FiniteTempBasisSet) = beta(bset.basis_f)
wmax(bset::FiniteTempBasisSet) = wmax(bset.basis_f)

function Base.getproperty(bset::FiniteTempBasisSet, d::Symbol)
    if d === :tau
        return getfield(bset, :smpl_tau_f).sampling_points
    elseif d === :wn_f
        return getfield(bset, :smpl_wn_f).sampling_points
    elseif d === :wn_b
        return getfield(bset, :smpl_wn_b).sampling_points
    elseif d === :sve_result
        return getfield(bset, :basis_f).sve_result
    else
        return getfield(bset, d)
    end
end

function Base.propertynames(::FiniteTempBasisSet, private::Bool=false)
    return (:tau, :wn_f, :wn_b, :sve_result, propertynames(FiniteTempBasisSet, private)...)
end

function Base.show(io::IO, b::FiniteTempBasisSet)
    return print(io, "FiniteTempBasisSet: beta=$(beta(b)), wmax=$(wmax(b))")
end
