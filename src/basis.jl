abstract type AbstractBasis end

Base.size(basis::AbstractBasis) = size(basis.u)
getbeta(basis::AbstractBasis) = basis.β
getstatistics(basis::AbstractBasis) = basis.statistics

"""
    DimensionlessBasis <: AbstractBasis

Intermediate representation (IR) basis in reduced variables.

For a continuation kernel `K` from real frequencies, `ω ∈ [-ωmax, ωmax]`, to
imaginary time, `τ ∈ [0, β]`, this type stores the truncated singular
value expansion or IR basis:

    K(x, y) ≈ sum(u[l](x) * s[l] * v[l](y) for l in range(L))

The functions are given in reduced variables, `x = 2τ/β - 1` and
`y = ω/ωmax`, which scales both sides to the interval `[-1, 1]`.  The
kernel then only depends on a cutoff parameter `Λ = β * ωmax`.

# Examples
The following example code assumes the spectral function is a single
pole at `x = 0.2`. We first compute an IR basis suitable for fermions and `β*W ≤ 42`. Then we get G(iw) on the first few Matsubara frequencies:

```julia-repl
julia> using SparseIR

julia> basis = DimensionlessBasis(fermion, 42);

julia> gl = basis.s .* basis.v(0.2);

julia> giw = transpose(basis.uhat([1, 3, 5, 7])) * gl
```

# Fields
- `u::PiecewiseLegendrePolyVector`: Set of IR basis functions on the reduced imaginary time (`x`) axis. These functions are stored as piecewise Legendre polynomials.

  To obtain the value of all basis functions at a point or a array of
  points `x`, you can call the function `u(x)`.  To obtain a single
  basis function, a slice or a subset `l`, you can use `u[l]`.

- `uhat::PiecewiseLegendreFTVector`: Set of IR basis functions on the Matsubara frequency (`wn`) axis.
These objects are stored as a set of Bessel functions.

  To obtain the value of all basis functions at a Matsubara frequency
  or a array of points `wn`, you can call the function `uhat(wn)`.
  Note that we expect reduced frequencies, which are simply even/odd
  numbers for bosonic/fermionic objects. To obtain a single basis
  function, a slice or a subset `l`, you can use `uhat[l]`.

- `s`: Vector of singular values of the continuation kernel

- `v::PiecewiseLegendrePolyVector`: Set of IR basis functions on the reduced real frequency (`y`) axis.
These functions are stored as piecewise Legendre polynomials.

  To obtain the value of all basis functions at a point or a array of
  points `y`, you can call the function `v(y)`.  To obtain a single
  basis function, a slice or a subset `l`, you can use `v[l]`.

See also [`FiniteTempBasis`](@ref) for a basis directly in time/frequency.
"""
struct DimensionlessBasis{K<:AbstractKernel,T<:AbstractFloat,S<:Statistics} <: AbstractBasis
    kernel::K
    u::PiecewiseLegendrePolyVector{T}
    uhat::PiecewiseLegendreFTVector{T}
    s::Vector{T}
    v::PiecewiseLegendrePolyVector{T}
    statistics::S
end

function Base.show(io::IO, a::DimensionlessBasis)
    return print(io, "DimensionlessBasis: statistics=$(getstatistics(a)), size=$(size(a))")
end

"""
    DimensionlessBasis(statistics, Λ, ε=nothing; kernel=LogisticKernel(Λ), sve_result=compute_sve(kernel; ε))

Construct an IR basis suitable for the given `statistics` and cutoff `Λ`.
"""
function DimensionlessBasis(
    statistics::Statistics, Λ, ε=nothing;
    kernel=LogisticKernel(Λ), sve_result=compute_sve(kernel; ε),
)
    u, s, v = sve_result
    size(u) == size(s) == size(v) || throw(DimensionMismatch("Mismatched shapes in SVE"))

    # The radius of convergence of the asymptotic expansion is Λ/2,
    # so for significantly larger frequencies we use the asymptotics,
    # since it has lower relative error.
    û = map(ui -> PiecewiseLegendreFT(ui, statistics,
                                      conv_radius(kernel)), u)
    return DimensionlessBasis(kernel, u, û, s, v, statistics)
end

"""
    Λ(basis)

Basis cutoff parameter `Λ = β * ωmax`.
"""
Λ(basis::DimensionlessBasis) = basis.kernel.Λ

function Base.getindex(basis::DimensionlessBasis, i)
    sve_result = basis.u[i], basis.s[i], basis.v[i]
    return DimensionlessBasis(basis.statistics, Λ(basis); kernel=basis.kernel, sve_result)
end

"""
    FiniteTempBasis <: AbstractBasis

Intermediate representation (IR) basis for given temperature.

For a continuation kernel `K` from real frequencies, `ω ∈ [-ωmax, ωmax]`, to
imaginary time, `τ ∈ [0, beta]`, this type stores the truncated singular
value expansion or IR basis:

    K(τ, ω) ≈ sum(u[l](τ) * s[l] * v[l](ω) for l in 1:L)

This basis is inferred from a reduced form by appropriate scaling of
the variables.

# Examples
The following example code assumes the spectral function is a single
pole at `ω = 2.5`. We first compute an IR basis suitable for fermions and `β = 10`, `W ≤ 4.2`. Then we get G(iw) on the first few Matsubara frequencies:

```julia-repl
julia> using SparseIR

julia> basis = FiniteTempBasis(fermion, 42, 4.2);

julia> gl = basis.s .* basis.v(2.5);

julia> giw = transpose(basis.uhat([1, 3, 5, 7])) * gl
```

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
struct FiniteTempBasis{K,T,S} <: AbstractBasis
    kernel::K
    sve_result::Tuple{
        PiecewiseLegendrePolyVector{T},Vector{T},PiecewiseLegendrePolyVector{T}
    }
    statistics::S
    β::T
    u::PiecewiseLegendrePolyVector{T}
    v::PiecewiseLegendrePolyVector{T}
    s::Vector{T}
    uhat::PiecewiseLegendreFTVector{T,S}
end

const _DEFAULT_FINITE_TEMP_BASIS = FiniteTempBasis{LogisticKernel,Float64}

function Base.show(io::IO, a::FiniteTempBasis{K,T}) where {K,T}
    return print(
        io, "FiniteTempBasis{$K, $T}($(getstatistics(a)), $(getbeta(a)), $(getwmax(a)))"
    )
end

"""
    FiniteTempBasis(statistics, β, wmax, ε=nothing; kernel=LogisticKernel(β * wmax), sve_result=compute_sve(kernel; ε))

Construct a finite temperature basis suitable for the given `statistics` and cutoffs `β` and `wmax`.
"""
function FiniteTempBasis(
    statistics::Statistics, β, wmax, ε=nothing;
    kernel=LogisticKernel(β * wmax), sve_result=compute_sve(kernel; ε),
)
    β > 0 || throw(DomainError(β, "Inverse temperature β must be positive"))
    wmax ≥ 0 || throw(DomainError(wmax, "Frequency cutoff wmax must be non-negative"))

    u, s, v = sve_result
    size(u) == size(s) == size(v) || throw(DimensionMismatch("Mismatched shapes in SVE"))

    # The polynomials are scaled to the new variables by transforming the
    # knots according to: tau = beta/2 * (x + 1), w = wmax * y.  Scaling
    # the data is not necessary as the normalization is inferred.
    wmax = kernel.Λ / β
    u_knots = β / 2 * (u.knots .+ 1)
    v_knots = wmax * v.knots
    u_ = PiecewiseLegendrePolyVector(u, u_knots; Δx=β / 2 * u.Δx, symm=u.symm)
    v_ = PiecewiseLegendrePolyVector(v, v_knots; Δx=wmax * v.Δx, symm=v.symm)

    # The singular values are scaled to match the change of variables, with
    # the additional complexity that the kernel may have an additional
    # power of w.
    s_ = √(β / 2 * wmax) * wmax^(-ypower(kernel)) * s

    # HACK: as we don't yet support Fourier transforms on anything but the
    # unit interval, we need to scale the underlying data.  This breaks
    # the correspondence between U.hat and Uhat though.
    û = map(ui -> PiecewiseLegendreFT(scale(ui, √β), statistics,
                                      conv_radius(kernel)), u)

    return FiniteTempBasis(kernel, sve_result, statistics, float(β), u_, v_, s_, û)
end

Base.firstindex(::AbstractBasis) = 1
Base.length(basis::AbstractBasis) = length(basis.s)

"""
    iswellconditioned(basis)

Return `true` if the sampling is expected to be well-conditioned.
"""
iswellconditioned(::DimensionlessBasis) = true
iswellconditioned(::FiniteTempBasis) = true

function Base.getindex(basis::FiniteTempBasis, i)
    u, s, v = basis.sve_result
    sve_result = u[i], s[i], v[i]
    return FiniteTempBasis(
        basis.statistics, getbeta(basis), getwmax(basis); kernel=basis.kernel, sve_result
    )
end

"""
    getwmax(basis::FiniteTempBasis)

Real frequency cutoff.
"""
getwmax(basis::FiniteTempBasis) = basis.kernel.Λ / getbeta(basis)

"""
    finite_temp_bases(β, wmax, ε, sve_result=compute_sve(LogisticKernel(β * wmax); ε))

Construct FiniteTempBasis objects for fermion and bosons using the same LogisticKernel instance.
"""
function finite_temp_bases(
    β::AbstractFloat, wmax::AbstractFloat, ε,
    sve_result=compute_sve(LogisticKernel(β * wmax); ε),
)
    basis_f = FiniteTempBasis(fermion, β, wmax, ε; sve_result)
    basis_b = FiniteTempBasis(boson, β, wmax, ε; sve_result)
    return basis_f, basis_b
end

"""
    default_tau_sampling_points(basis)

Default sampling points on the imaginary time/`x` axis.
"""
default_tau_sampling_points(basis::AbstractBasis) = _default_sampling_points(basis.u)

"""
    default_matsubara_sampling_points(basis; mitigate=true)

Default sampling points on the imaginary frequency axis.
"""
function default_matsubara_sampling_points(basis::AbstractBasis; mitigate=true)
    return _default_matsubara_sampling_points(basis.uhat, mitigate)
end

"""
    default_omega_sampling_points(basis)

Default sampling points on the real-frequency axis.
"""
default_omega_sampling_points(basis::AbstractBasis) = _default_sampling_points(basis.v)

function _default_sampling_points(u)
    poly = last(u)
    maxima = roots(deriv(poly))
    left = (first(maxima) + poly.xmin) / 2
    right = (last(maxima) + poly.xmax) / 2
    return [left; maxima; right]
end

function _default_matsubara_sampling_points(uhat, mitigate=true)
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
    if uhat.stat == Bosonic()
        pushfirst!(wn, 0)
        sort!(wn)
        unique!(wn)
    end

    return wn
end

function _get_kernel(Λ, kernel)
    if isnothing(kernel)
        kernel = LogisticKernel(Λ)
    elseif kernel.Λ ≉ Λ
        error("kernel.Λ ≉ Λ")
    end
    return kernel
end
