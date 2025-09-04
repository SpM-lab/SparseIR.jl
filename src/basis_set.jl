"""
    FiniteTempBasisSet

Type for holding IR bases and sparse-sampling objects.

An object of this type holds IR bases for fermions and bosons
and associated sparse-sampling objects.

# Fields

  - basis_f::FiniteTempBasis: Fermion basis
  - basis_b::FiniteTempBasis: Boson basis
  - tau::Vector{Float64}: Sampling points in the imaginary-time domain
  - wn_f::Vector{Int}: Sampling fermionic frequencies
  - wn_b::Vector{Int}: Sampling bosonic frequencies
  - smpl_tau_f::TauSampling: Sparse sampling for tau & fermion
  - smpl_tau_b::TauSampling: Sparse sampling for tau & boson
  - smpl_wn_f::MatsubaraSampling: Sparse sampling for Matsubara frequency & fermion
  - smpl_wn_b::MatsubaraSampling: Sparse sampling for Matsubara frequency & boson
  - sve_result::Tuple{PiecewiseLegendrePoly,Vector{Float64},PiecewiseLegendrePoly}: Results of SVE

# Getters

  - beta::Float64: Inverse temperature
  - ωmax::Float64: Cut-off frequency
"""
struct FiniteTempBasisSet
    basis_f    :: FiniteTempBasis{Fermionic}
    basis_b    :: FiniteTempBasis{Bosonic}
    smpl_tau_f :: TauSampling64F
    smpl_tau_b :: TauSampling64B
    smpl_wn_f  :: MatsubaraSampling64F
    smpl_wn_b  :: MatsubaraSampling64B

    """
        FiniteTempBasisSet(β, ωmax[, ε]; [sve_result])

    Create basis sets for fermion and boson and associated sampling objects.
    Fermion and bosonic bases are constructed by SVE of the logistic kernel.
    """
    function FiniteTempBasisSet(β::Real, ωmax::Real, ε::Real;
                                sve_result=SVEResult(LogisticKernel(β * ωmax), ε))
        basis_f = FiniteTempBasis{Fermionic}(β, ωmax, ε; sve_result)
        basis_b = FiniteTempBasis{Bosonic}(β, ωmax, ε; sve_result)

        tau_sampling_f = TauSampling(basis_f)
        tau_sampling_b = TauSampling(basis_b)
        matsubara_sampling_f = MatsubaraSampling(basis_f)
        matsubara_sampling_b = MatsubaraSampling(basis_b)

        new(basis_f, basis_b, tau_sampling_f, tau_sampling_b,
            matsubara_sampling_f, matsubara_sampling_b)
    end
end

β(bset::FiniteTempBasisSet) = β(bset.basis_f)
ωmax(bset::FiniteTempBasisSet) = ωmax(bset.basis_f)

function Base.getproperty(bset::FiniteTempBasisSet, d::Symbol)
    if d === :tau
        return sampling_points(bset.smpl_tau_f)
    elseif d === :wn_f
        return sampling_points(bset.smpl_wn_f)
    elseif d === :wn_b
        return sampling_points(bset.smpl_wn_b)
    elseif d === :sve_result
        return sve_result(bset.basis_f)
    else
        return getfield(bset, d)
    end
end

function Base.propertynames(::FiniteTempBasisSet)
    (:tau, :wn_f, :wn_b, :sve_result, fieldnames(FiniteTempBasisSet)...)
end

function Base.show(io::IO, ::MIME"text/plain", b::FiniteTempBasisSet)
    print(io, "FiniteTempBasisSet with β = $(beta(b)), ωmax = $(wmax(b))")
end
