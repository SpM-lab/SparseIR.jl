"""
    FiniteTempBasis <: AbstractBasis

Intermediate representation (IR) basis for given temperature.

For a continuation kernel `K` from real frequencies, `ω ∈ [-ωmax, ωmax]`, to
imaginary time, `τ ∈ [0, beta]`, this type stores the truncated singular
value expansion or IR basis:

    K(τ, ω) ≈ sum(u[l](τ) * s[l] * v[l](ω) for l in 1:L)

This basis is inferred from a reduced form by appropriate scaling of
the variables.

# Fields

  - `u::PiecewiseLegendrePolyVector`:
    Set of IR basis functions on the imaginary time (`tau`) axis.
    These functions are stored as piecewise Legendre polynomials.
    
    To obtain the value of all basis functions at a point or a array of
    points `x`, you can call the function `u(x)`.  To obtain a single
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
    points `w`, you can call the function `v(w)`.  To obtain a single
    basis function, a slice or a subset `l`, you can use `v[l]`.
"""
struct FiniteTempBasis{K,T,S,TP} <: AbstractBasis
    kernel     :: K
    sve_result :: SVEResult{T, K, TP}
    statistics :: S
    accuracy   :: T
    β          :: T
    u          :: PiecewiseLegendrePolyVector{T}
    v          :: PiecewiseLegendrePolyVector{T}
    s          :: Vector{T}
    uhat       :: PiecewiseLegendreFTVector{T,S}
end

"""
    FiniteTempBasis(statistics, β, ωmax, ε=nothing;
                    kernel=LogisticKernel(β * ωmax), sve_result=compute_sve(kernel; ε))

Construct a finite temperature basis suitable for the given `statistics` and
cutoffs `β` and `ωmax`.
"""
function FiniteTempBasis(statistics::Statistics, β::Number, ωmax::Number, ε=nothing;
                         max_size = typemax(Int), kernel=LogisticKernel(β * ωmax), 
                         sve_result=compute_sve(kernel; ε))
    β > 0 || throw(DomainError(β, "Inverse temperature β must be positive"))
    ωmax ≥ 0 || throw(DomainError(ωmax, "Frequency cutoff ωmax must be non-negative"))

    u, s, v = isnothing(ε) ? part(sve_result; max_size) : part(sve_result; ε, max_size)
    if length(sve_result.s) > length(s)
        accuracy = sve_result.s[length(s) + 1] / first(sve_result.s)
    else
        accuracy = last(sve_result.s) / first(sve_result.s)
    end

    # The polynomials are scaled to the new variables by transforming the
    # knots according to: tau = beta/2 * (x + 1), w = ωmax * y.  Scaling
    # the data is not necessary as the normalization is inferred.
    ωmax = kernel.Λ / β
    u_knots = β / 2 * (u.knots .+ 1)
    v_knots = ωmax * v.knots
    u_ = PiecewiseLegendrePolyVector(u, u_knots; Δx=β / 2 * u.Δx, symm=u.symm)
    v_ = PiecewiseLegendrePolyVector(v, v_knots; Δx=ωmax * v.Δx, symm=v.symm)

    # The singular values are scaled to match the change of variables, with
    # the additional complexity that the kernel may have an additional
    # power of w.
    s_ = √(β / 2 * ωmax) * ωmax^(-ypower(kernel)) * s

    # HACK: as we don't yet support Fourier transforms on anything but the
    # unit interval, we need to scale the underlying data.  This breaks
    # the correspondence between U.hat and Uhat though.
    û = map(ui -> PiecewiseLegendreFT(scale(ui, √β), statistics, conv_radius(kernel)), u)

    return FiniteTempBasis(kernel, sve_result, statistics, accuracy, float(β), u_, v_, s_, û)
end

const _DEFAULT_FINITE_TEMP_BASIS = FiniteTempBasis{LogisticKernel,Float64}

function Base.show(io::IO, a::FiniteTempBasis{K,T}) where {K,T}
    print(io, "FiniteTempBasis{$K, $T}($(statistics(a)), $(beta(a)), $(ωmax(a)))")
end

Base.getindex(basis::FiniteTempBasis, i) =
    FiniteTempBasis(statistics(basis), beta(basis), ωmax(basis), nothing;
                    max_size=range_to_size(i), kernel=basis.kernel, sve_result=basis.sve_result)

significance(basis::FiniteTempBasis) = basis.s ./ first(basis.s)
accuracy(basis::FiniteTempBasis) = basis.accuracy
ωmax(basis::FiniteTempBasis) = basis.kernel.Λ / beta(basis)
sve_result(basis::FiniteTempBasis) = basis.sve_result
kernel(basis::FiniteTempBasis) = basis.kernel

default_tau_sampling_points(basis::FiniteTempBasis) = default_sampling_points(basis.u)
default_matsubara_sampling_points(basis::FiniteTempBasis; mitigate=true) = 
    default_matsubara_sampling_points(basis.uhat, mitigate)
default_omega_sampling_points(basis::FiniteTempBasis) = default_sampling_points(basis.v)


"""
    rescale(basis::FiniteTempBasis, new_β)

Return a basis for different temperature.

Uses the same kernel with the same ``ε``, but a different
temperature.  Note that this implies a different UV cutoff ``ωmax``,
since ``Λ == β * ωmax`` stays constant.
"""
function rescale(basis::FiniteTempBasis, new_β)
    new_ωmax = Λ(kernel(basis)) / new_β
    return FiniteTempBasis(statistics(basis), new_β, new_ωmax, nothing;
                        max_size=length(basis), kernel=kernel(basis),
                        sve_result=sve_result(basis))
end

"""
    finite_temp_bases(β, ωmax, ε, sve_result=compute_sve(LogisticKernel(β * ωmax); ε))

Construct `FiniteTempBasis` objects for fermion and bosons using the same
`LogisticKernel` instance.
"""
function finite_temp_bases(β::AbstractFloat, ωmax::AbstractFloat, ε,
                           sve_result=compute_sve(LogisticKernel(β * ωmax); ε))
    basis_f = FiniteTempBasis(fermion, β, ωmax, ε; sve_result)
    basis_b = FiniteTempBasis(boson, β, ωmax, ε; sve_result)
    return basis_f, basis_b
end

function default_sampling_points(u)
    poly = last(u)
    maxima = roots(deriv(poly))
    left = (first(maxima) + poly.xmin) / 2
    right = (last(maxima) + poly.xmax) / 2
    return [left; maxima; right]
end

function default_matsubara_sampling_points(uhat::PiecewiseLegendreFTVector, mitigate=true)
    # Use the (discrete) extrema of the corresponding highest-order basis
    # function in Matsubara.  This turns out to be close to optimal with
    # respect to conditioning for this size (within a few percent).
    polyhat = last(uhat)
    wn = findextrema(polyhat)

    # While the condition number for sparse sampling in tau saturates at a
    # modest level, the conditioning in Matsubara steadily deteriorates due
    # to the fact that we are not free to set sampling points continuously.
    # At double precision, tau sampling is better conditioned than iwn
    # by a factor of ~4 (still OK). To battle this, we fence the largest
    # frequency with two carefully chosen oversampling points, which brings
    # the two sampling problems within a factor of 2.
    if mitigate
        for wn_max in (first(wn), last(wn))
            wn_diff = BosonicFreq(2 * round(Int, 0.025 * Integer(wn_max)))
            length(wn) ≥ 20 && push!(wn, wn_max - sign(wn_max) * wn_diff)
            length(wn) ≥ 42 && push!(wn, wn_max + sign(wn_max) * wn_diff)
        end
        sort!(wn)
        unique!(wn)
    end

    # For bosonic function
    if statistics(uhat) isa Bosonic
        pushfirst!(wn, 0)
        sort!(wn)
        unique!(wn)
    end

    return wn
end

function range_to_size(range::UnitRange)
    isone(first(range)) || error("Range must start at 1.")
    return last(range)
end
