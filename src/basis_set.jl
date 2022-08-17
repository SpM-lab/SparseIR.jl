"""
    FiniteTempBasisSet

Type for holding IR bases and sparse-sampling objects.

An object of this type holds IR bases for fermions and bosons
and associated sparse-sampling objects.

# Fields

  - basis_f::FiniteTempBasis: Fermion basis
  - basis_b::FiniteTempBasis: Boson basis
  - beta::Float64: Inverse temperature
  - ωmax::Float64: Cut-off frequency
  - tau::Vector{Float64}: Sampling points in the imaginary-time domain
  - wn_f::Vector{Int}: Sampling fermionic frequencies
  - wn_b::Vector{Int}: Sampling bosonic frequencies
  - smpl_tau_f::TauSampling: Sparse sampling for tau & fermion
  - smpl_tau_b::TauSampling: Sparse sampling for tau & boson
  - smpl_wn_f::MatsubaraSampling: Sparse sampling for Matsubara frequency & fermion
  - smpl_wn_b::MatsubaraSampling: Sparse sampling for Matsubara frequency & boson
  - sve_result::Tuple{PiecewiseLegendrePoly,Vector{Float64},PiecewiseLegendrePoly}: Results of SVE
"""
struct FiniteTempBasisSet{TSF<:TauSampling,MSF<:MatsubaraSampling,TSB<:TauSampling,
                          MSB<:MatsubaraSampling}
    basis_f    :: DEFAULT_FINITE_TEMP_BASIS{Fermionic}
    basis_b    :: DEFAULT_FINITE_TEMP_BASIS{Bosonic}
    smpl_tau_f :: TSF
    smpl_tau_b :: TSB
    smpl_wn_f  :: MSF
    smpl_wn_b  :: MSB

    """
        FiniteTempBasisSet(β, ωmax, ε; sve_result=SVEResult(LogisticKernel(β * ωmax); ε))

    Create basis sets for fermion and boson and
    associated sampling objects.
    Fermion and bosonic bases are constructed by SVE of the logistic kernel.
    """
    function FiniteTempBasisSet(β::AbstractFloat, ωmax::AbstractFloat, ε;
                                sve_result=SVEResult(LogisticKernel(β * ωmax); ε))
        # Create bases using the given sve results
        basis_f = FiniteTempBasis(Fermionic(), β, ωmax, ε; sve_result)
        basis_b = FiniteTempBasis(Bosonic(), β, ωmax, ε; sve_result)

        tau_sampling_f = TauSampling(basis_f)
        tau_sampling_b = TauSampling(basis_b)
        matsubara_sampling_f = MatsubaraSampling(basis_f)
        matsubara_sampling_b = MatsubaraSampling(basis_b)

        new{typeof(tau_sampling_f),typeof(matsubara_sampling_f),
            typeof(tau_sampling_b),typeof(matsubara_sampling_b)}(basis_f, basis_b,
                                                                 tau_sampling_f,
                                                                 tau_sampling_b,
                                                                 matsubara_sampling_f,
                                                                 matsubara_sampling_b)
    end
end

β(bset::FiniteTempBasisSet) = β(bset.basis_f)
ωmax(bset::FiniteTempBasisSet) = ωmax(bset.basis_f)

function Base.getproperty(bset::FiniteTempBasisSet, d::Symbol)
    if d === :tau
        # return getfield(bset, :smpl_tau_f).sampling_points
        return bset.smpl_tau_f.sampling_points
    elseif d === :wn_f
        # return getfield(bset, :smpl_wn_f).sampling_points
        return bset.smpl_wn_f.sampling_points
    elseif d === :wn_b
        # return getfield(bset, :smpl_wn_b).sampling_points
        return bset.smpl_wn_b.sampling_points
    elseif d === :sve_result
        # return getfield(bset, :basis_f).sve_result
        return bset.basis_f.sve_result
    else
        return getfield(bset, d)
    end
end

function Base.propertynames(::FiniteTempBasisSet, private::Bool=false)
    return (:tau, :wn_f, :wn_b, :sve_result, propertynames(FiniteTempBasisSet, private)...)
end

function Base.show(io::IO, b::FiniteTempBasisSet)
    return print(io, "FiniteTempBasisSet: beta=$(beta(b)), ωmax=$(ωmax(b))")
end
