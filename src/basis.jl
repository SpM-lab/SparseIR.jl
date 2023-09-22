"""
    FiniteTempBasis <: AbstractBasis

Intermediate representation (IR) basis for given temperature.

For a continuation kernel `K` from real frequencies, `ω ∈ [-ωmax, ωmax]`, to
imaginary time, `τ ∈ [0, β]`, this type stores the truncated singular
value expansion or IR basis:

    K(τ, ω) ≈ sum(u[l](τ) * s[l] * v[l](ω) for l in 1:L)

This basis is inferred from a reduced form by appropriate scaling of
the variables.

# Fields

  - `u::PiecewiseLegendrePolyVector`:
    Set of IR basis functions on the imaginary time (`tau`) axis.
    These functions are stored as piecewise Legendre polynomials.

    To obtain the value of all basis functions at a point or a array of
    points `x`, you can call the function `u(x)`. To obtain a single
    basis function, a slice or a subset `l`, you can use `u[l]`.

  - `uhat::PiecewiseLegendreFT`:
    Set of IR basis functions on the Matsubara frequency (`wn`) axis.
    These objects are stored as a set of Bessel functions.

    To obtain the value of all basis functions at a Matsubara frequency
    or a array of points `wn`, you can call the function `uhat(wn)`.
    Note that we expect reduced frequencies, which are simply even/odd
    numbers for bosonic/fermionic objects. To obtain a single basis
    function, a slice or a subset `l`, you can use `uhat[l]`.
  - `s`: Vector of singular values of the continuation kernel
  - `v::PiecewiseLegendrePoly`:
    Set of IR basis functions on the real frequency (`w`) axis.
    These functions are stored as piecewise Legendre polynomials.

    To obtain the value of all basis functions at a point or a array of
    points `w`, you can call the function `v(w)`. To obtain a single
    basis function, a slice or a subset `l`, you can use `v[l]`.
"""
struct FiniteTempBasis{S,K} <: AbstractBasis{S}
    kernel     :: K
    sve_result :: SVEResult{K}
    accuracy   :: Float64
    β          :: Float64
    u          :: PiecewiseLegendrePolyVector
    v          :: PiecewiseLegendrePolyVector
    s          :: Vector{Float64}
    uhat       :: PiecewiseLegendreFTVector{S}
    uhat_full  :: PiecewiseLegendreFTVector{S}
end

"""
    FiniteTempBasis{S}(β, ωmax, ε=nothing; max_size=nothing, args...)

Construct a finite temperature basis suitable for the given `S` (`Fermionic`
or `Bosonic`) and cutoffs `β` and `ωmax`.
"""
function FiniteTempBasis{S}(β::Real, ωmax::Real, ε=nothing; max_size=nothing,
                            kernel=LogisticKernel(β * ωmax),
                            sve_result=SVEResult(kernel; ε)) where {S}
    FiniteTempBasis(S(), β, ωmax, ε; max_size, kernel, sve_result)
end

function FiniteTempBasis(statistics::Statistics, β::Real, ωmax::Real, ε=nothing;
                         max_size=nothing, kernel=LogisticKernel(β * ωmax),
                         sve_result=SVEResult(kernel; ε))
    β > 0 || throw(DomainError(β, "Inverse temperature β must be positive"))
    ωmax ≥ 0 || throw(DomainError(ωmax, "Frequency cutoff ωmax must be non-negative"))

    u_, s_, v_ = part(sve_result; ε, max_size)

    accuracy = if length(sve_result.s) > length(s_)
        sve_result.s[length(s_) + 1] / first(sve_result.s)
    else
        last(sve_result.s) / first(sve_result.s)
    end

    # The polynomials are scaled to the new variables by transforming the
    # knots according to: tau = β/2 * (x + 1), w = ωmax * y. Scaling
    # the data is not necessary as the normalization is inferred.
    ωmax = Λ(kernel) / β
    u_knots = (β / 2) .* (u_.knots .+ 1)
    v_knots = ωmax .* v_.knots
    u = PiecewiseLegendrePolyVector(u_, u_knots; Δx=(β / 2) .* u_.Δx, symm=u_.symm)
    v = PiecewiseLegendrePolyVector(v_, v_knots; Δx=ωmax .* v_.Δx, symm=v_.symm)

    # The singular values are scaled to match the change of variables, with
    # the additional complexity that the kernel may have an additional
    # power of w.
    s = (sqrt(β / 2 * ωmax) * ωmax^(-ypower(kernel))) .* s_

    # HACK: as we don't yet support Fourier transforms on anything but the
    # unit interval, we need to scale the underlying data.
    û_base_full = PiecewiseLegendrePolyVector(sqrt(β) .* sve_result.u.data, sve_result.u)
    û_full = PiecewiseLegendreFTVector(û_base_full, statistics;
                                        n_asymp=conv_radius(kernel))
    û = û_full[1:length(s)]

    return FiniteTempBasis(kernel, sve_result, accuracy, float(β), u, v, s, û, û_full)
end

function Base.show(io::IO, ::MIME"text/plain", b::FiniteTempBasis)
    print(io, "$(length(b))-element FiniteTempBasis{$(typeof(statistics(b)))} with ")
    println(io, "β = $(β(b)), ωmax = $(ωmax(b)) and singular values:")
    for s in b.s[begin:(end - 1)]
        println(io, " $s")
    end
    print(io, " $(last(b.s))")
end

function Base.getindex(basis::FiniteTempBasis, i::AbstractRange)
    FiniteTempBasis(statistics(basis), β(basis), ωmax(basis), nothing;
                    max_size=range_to_length(i), kernel=basis.kernel,
                    sve_result=basis.sve_result)
end

significance(basis::FiniteTempBasis) = basis.s ./ first(basis.s)
accuracy(basis::FiniteTempBasis) = basis.accuracy
ωmax(basis::FiniteTempBasis) = Λ(basis) / β(basis)
sve_result(basis::FiniteTempBasis) = basis.sve_result
kernel(basis::FiniteTempBasis) = basis.kernel
Λ(basis::FiniteTempBasis) = Λ(kernel(basis))

function default_tau_sampling_points(basis::FiniteTempBasis)
    x = default_sampling_points(basis.sve_result.u, length(basis))
    β(basis) / 2 * (x .+ 1)
end
function default_matsubara_sampling_points(basis::FiniteTempBasis; positive_only=false)
    default_matsubara_sampling_points(basis.uhat_full, length(basis); positive_only)
end
function default_omega_sampling_points(basis::FiniteTempBasis)
    y = default_sampling_points(basis.sve_result.v, length(basis))
    ωmax(basis) * y
end

"""
    rescale(basis::FiniteTempBasis, new_β)

Return a basis for different temperature.

Uses the same kernel with the same ``ε``, but a different
temperature. Note that this implies a different UV cutoff ``ωmax``,
since ``Λ == β * ωmax`` stays constant.
"""
function rescale(basis::FiniteTempBasis, new_β)
    new_ωmax = Λ(kernel(basis)) / new_β
    return FiniteTempBasis(statistics(basis), new_β, new_ωmax, nothing;
                           max_size=length(basis), kernel=kernel(basis),
                           sve_result=sve_result(basis))
end

"""
    finite_temp_bases(β::Real, ωmax::Real, ε=nothing;
                      kernel=LogisticKernel(β * ωmax), sve_result=SVEResult(kernel; ε))

Construct `FiniteTempBasis` objects for fermion and bosons using the same
`LogisticKernel` instance.
"""
function finite_temp_bases(β::Real, ωmax::Real, ε=nothing;
                           kernel=LogisticKernel(β * ωmax),
                           sve_result=SVEResult(kernel; ε))
    basis_f = FiniteTempBasis{Fermionic}(β, ωmax, ε; sve_result)
    basis_b = FiniteTempBasis{Bosonic}(β, ωmax, ε; sve_result)
    return basis_f, basis_b
end

function default_sampling_points(u::PiecewiseLegendrePolyVector, L::Integer)
    (u.xmin, u.xmax) == (-1, 1) || error("Expecting unscaled functions here.")

    # For orthogonal polynomials (the high-T limit of IR), we know that the
    # ideal sampling points for a basis of size L are the roots of the L-th
    # polynomial. We empirically find that these stay good sampling points
    # for our kernels (probably because the kernels are totally positive).
    if L < length(u)
        x₀ = roots(u[L + 1])
    else
        # If we do not have enough polynomials in the basis, we approximate the
        # roots of the L'th polynomial by the extrema of the last basis
        # function, which is sensible due to the strong interleaving property
        # of these functions' roots.
        poly = last(u)
        maxima = roots(deriv(poly))

        # Putting the sampling points right at [0, β], which would be the
        # local extrema, is slightly worse conditioned than putting it in the
        # middel. This can be understood by the fact that the roots never
        # occur right at the border.
        left  = (first(maxima) + poly.xmin) / 2
        right = (last(maxima) + poly.xmax) / 2
        x₀    = [left; maxima; right]
    end

    length(x₀) == L || @warn """
        Expecting to get $L sampling points for corresponding basis function,
        instead got $(length(x₀)). This may happen if not enough precision is
        left in the polynomial.
        """
    return x₀
end

function default_matsubara_sampling_points(û::PiecewiseLegendreFTVector, L::Integer;
                                           fence=false, positive_only=false)
    l_requested = L

    # The number of sign changes is always odd for bosonic basis and even for fermionic 
    # basis. So in order to get at least as many sign changes as basis functions:
    statistics(û) isa Fermionic && isodd(l_requested) && (l_requested += 1)
    statistics(û) isa Bosonic && iseven(l_requested) && (l_requested += 1)

    # As with the zeros, the sign changes provide excellent sampling points
    if l_requested < length(û)
        ωn = sign_changes(û[l_requested + 1]; positive_only)
    else
        # As a fallback, use the (discrete) extrema of the corresponding
        # highest-order basis function in Matsubara. This turns out to be okay.
        ωn = find_extrema(last(û); positive_only)

        # For bosonic bases, we must explicitly include the zero frequency,
        # otherwise the condition number blows up.
        if statistics(û) isa Bosonic
            pushfirst!(ωn, 0)
            sort!(ωn)
            unique!(ωn)
        end
    end

    expected_size = l_requested
    positive_only && (expected_size = div(expected_size + 1, 2))

    length(ωn) == expected_size || @warn """
        Requesting $expected_size $(statistics(û)) sampling frequencies for basis size
        L = $L, but $(length(ωn)) were returned. This may indicate a problem with precision.
        """

    fence && fence_matsubara_sampling!(ωn, positive_only)
    return ωn
end

function fence_matsubara_sampling!(ωn::Vector{<:MatsubaraFreq}, positive_only)
    # While the condition number for sparse sampling in tau saturates at a
    # modest level, the conditioning in Matsubara steadily deteriorates due
    # to the fact that we are not free to set sampling points continuously.
    # At double precision, tau sampling is better conditioned than iwn
    # by a factor of ~4 (still OK). To battle this, we fence the largest
    # frequency with two carefully chosen oversampling points, which brings
    # the two sampling problems within a factor of 2.
    outer_frequencies = positive_only ? (last(ωn),) : (first(ωn), last(ωn))
    for ωn_outer in outer_frequencies
        ωn_diff = BosonicFreq(2 * round(Int, 0.025 * Int(ωn_outer)))
        length(ωn) ≥ 20 && push!(ωn, ωn_outer - sign(ωn_outer) * ωn_diff)
        length(ωn) ≥ 42 && push!(ωn, ωn_outer + sign(ωn_outer) * ωn_diff)
    end
    return unique!(ωn)
end

function range_to_length(range::UnitRange)
    isone(first(range)) || error("Range must start at 1.")
    return last(range)
end

# """
#     evaluate(basis::FiniteTempBasis, a::Vector, τ::Real)

# Compute ``\\sum_l a_l U_l(τ)``.
# """
# function evaluate(basis::FiniteTempBasis, a::Vector, τ::Real)
#     length(basis.u) == length(a) || error("wrong number of coefficients")
#     0 <= τ <= basis.β || error("τ is out of bounds")
#     sum(zip(a, basis.u); init=zero(eltype(a))) do (aₗ, uₗ)
#         aₗ * uₗ(τ)
#     end
# end

# """
#     evaluate(basis::FiniteTempBasis{S}, a::Vector, iω::MatsubaraFreq{S}) where {S}

# Compute ``\\sum_l a_l \\hat{U}_l(iω)``.
# """
# function evaluate(basis::FiniteTempBasis{S}, a::Vector, iω::MatsubaraFreq{S}) where {S}
#     length(basis.uhat) == length(a) || error("wrong number of coefficients")
#     û_iω = basis.uhat(iω)
#     û_iω .*= a
#     sum(û_iω)
# end

# evaluate(basis::FiniteTempBasis, a::Vector, x::AbstractArray) = evaluate.(basis, Ref(a), x)
