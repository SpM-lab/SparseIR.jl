export IRBasis, FiniteTempBasis, finite_temp_bases, fermion, boson

@enum Statistics fermion boson

abstract type AbstractBasis end

function Base.getproperty(obj::AbstractBasis, d::Symbol)
    if d === :size
        return length(getfield(obj, :u))
    elseif d === :beta # backward compatibility
        return getfield(obj, :β)
    else
        return getfield(obj, d)
    end
end

Base.size(basis::AbstractBasis) = length(basis.u)
beta(basis::AbstractBasis) = basis.β

"""
    IRBasis <: AbstractBasis

Intermediate representation (IR) basis in reduced variables.

For a continuation kernel `K` from real frequencies, `ω ∈ [-ωmax, ωmax]`, to
imaginary time, `τ ∈ [0, β]`, this class stores the truncated singular
value expansion or IR basis:

    K(x, y) ≈ sum(u[l](x) * s[l] * v[l](y) for l in range(L))

The functions are given in reduced variables, `x = 2τ/β - 1` and
`y = ω/ωmax`, which scales both sides to the interval `[-1, 1]`.  The
kernel then only depends on a cutoff parameter `Λ = β * ωmax`.

# Examples
The following example code assumes the spectral function is a single
pole at `x = 0.2`. We first compute an IR basis suitable for fermions and `β*W <= 42`. Then we get G(iw) on the first few Matsubara frequencies:

```julia-repl
julia> using SparseIR

julia> basis = IRBasis(fermion, 42);

julia> gl = basis.s .* basis.v(0.2);

julia> giw = transpose(basis.uhat([1, 3, 5, 7])) * gl
```

# Fields
- `u::PiecewiseLegendrePolyArray`: Set of IR basis functions on the reduced imaginary time (`x`) axis. These functions are stored as piecewise Legendre polynomials.

  To obtain the value of all basis functions at a point or a array of
  points `x`, you can call the function `u(x)`.  To obtain a single
  basis function, a slice or a subset `l`, you can use `u[l]`.

- `uhat::PiecewiseLegendreFTArray`: Set of IR basis functions on the Matsubara frequency (`wn`) axis.
These objects are stored as a set of Bessel functions.

  To obtain the value of all basis functions at a Matsubara frequency
  or a array of points `wn`, you can call the function `uhat(wn)`.
  Note that we expect reduced frequencies, which are simply even/odd
  numbers for bosonic/fermionic objects. To obtain a single basis
  function, a slice or a subset `l`, you can use `uhat[l]`.

- `s`: Vector of singular values of the continuation kernel

- `v::PiecewiseLegendrePolyArray`: Set of IR basis functions on the reduced real frequency (`y`) axis.
These functions are stored as piecewise Legendre polynomials.

  To obtain the value of all basis functions at a point or a array of
  points `y`, you can call the function `v(y)`.  To obtain a single
  basis function, a slice or a subset `l`, you can use `v[l]`.

See also [`FiniteTempBasis`](@ref) for a basis directly in time/frequency.
"""
struct IRBasis{K<:AbstractKernel,T<:AbstractFloat} <: AbstractBasis
    kernel::K
    u::PiecewiseLegendrePolyArray{T}
    uhat::PiecewiseLegendreFTArray{T}
    s::Vector{T}
    v::PiecewiseLegendrePolyArray{T}
    sampling_points_v::Vector{T}
    statistics::Statistics
end

"""
    IRBasis(statistics, Λ, ε=nothing; kernel=nothing, sve_result=nothing)

Construct an IR basis suitable for the given `statistics` and cutoff `Λ`.
"""
function IRBasis(statistics, Λ, ε=nothing; kernel=nothing, sve_result=nothing)
    Λ = float(Λ)
    Λ ≥ 0 || error("Kernel cutoff Λ must be non-negative")

    self_kernel = _get_kernel(statistics, Λ, kernel)
    if isnothing(sve_result)
        u, s, v = compute(self_kernel; ε)
    else
        u, s, v = sve_result
        size(u) == size(s) == size(v) || error("Mismatched shapes in SVE")
    end

    if isnothing(ε) && isnothing(sve_result) && !HAVE_XPREC
        @warn """No extended precision is being used.
        Expect single precision (1.5e-8) only as both cutoff
        and accuracy of the basis functions."""
    end

    # The radius of convergence of the asymptotic expansion is Λ/2,
    # so for significantly larger frequencies we use the asymptotics,
    # since it has lower relative error.
    even_odd = Dict(fermion => :odd, boson => :even)[statistics]
    uhat = hat.(u, even_odd, 0:length(u)-1; n_asymp=conv_radius(self_kernel))
    rts = roots(last(v))
    sampling_points_v = [v.xmin; (rts[begin:(end - 1)] .+ rts[(begin + 1):end]) / 2; v.xmax]
    return IRBasis(self_kernel, u, uhat, s, v, sampling_points_v, statistics)
end

"""
    Λ(basis)

Basis cutoff parameter `Λ = β * ωmax`.
"""
Λ(basis::IRBasis) = basis.kernel.Λ

function Base.getindex(basis::IRBasis, i)
    sve_result = basis.u[i], basis.s[i], basis.v[i]
    return IRBasis(basis.statistics, Λ(basis); kernel=basis.kernel, sve_result)
end


"""
    FiniteTempBasis <: AbstractBasis

Intermediate representation (IR) basis for given temperature.

For a continuation kernel `K` from real frequencies, `ω ∈ [-ωmax, ωmax]`, to
imaginary time, `τ ∈ [0, beta]`, this class stores the truncated singular
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
- `u::PiecewiseLegendrePolyArray`:
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
struct FiniteTempBasis{K,T} <: AbstractBasis
    kernel::K
    sve_result::Tuple{PiecewiseLegendrePolyArray{T},Vector{T},PiecewiseLegendrePolyArray{T}}
    statistics::Statistics
    β::T
    u::PiecewiseLegendrePolyArray{T}
    v::PiecewiseLegendrePolyArray{T}
    s::Vector{T}
    uhat::PiecewiseLegendreFTArray{T}
end

"""
    FiniteTempBasis(statistics, β, wmax, ε=nothing; kernel=nothing, sve_result=nothing)

Construct a finite temperature basis suitable for the given `statistics` and cutoffs `β` and `wmax`.
"""
function FiniteTempBasis(statistics, β, wmax, ε=nothing; kernel=nothing, sve_result=nothing)
    β > 0 || error("Inverse temperature β must be positive")
    wmax ≥ 0 || error("Frequency cutoff wmax must be non-negative")

    kernel = _get_kernel(statistics, β * wmax, kernel)
    if isnothing(sve_result)
        u, s, v = compute(kernel; ε)
    else
        u, s, v = sve_result
        size(u) == size(s) == size(v) || error("Mismatched shapes in SVE")
    end

    if isnothing(ε) && isnothing(sve_result) && !HAVE_XPREC
        @warn """No extended precision is being used.
        Expect single precision (1.5e-8) only as both cutoff
        and accuracy of the basis functions."""
    end

    # The polynomials are scaled to the new variables by transforming the
    # knots according to: tau = beta/2 * (x + 1), w = wmax * y.  Scaling
    # the data is not necessary as the normalization is inferred.
    wmax = kernel.Λ / β
    u_ = PiecewiseLegendrePolyArray(u, β / 2 * (u.knots .+ 1); dx=β / 2 * u.dx, symm=u.symm)
    v_ = PiecewiseLegendrePolyArray(v, wmax * v.knots; dx=wmax * v.dx, symm=v.symm)

    # The singular values are scaled to match the change of variables, with
    # the additional complexity that the kernel may have an additional
    # power of w.
    s_ = √(β / 2 * wmax) * wmax^(-ypower(kernel)) * s

    # HACK: as we don't yet support Fourier transforms on anything but the
    # unit interval, we need to scale the underlying data.  This breaks
    # the correspondence between U.hat and Uhat though.
    û_base = scale.(u, √β)

    conv_radius = 40 * kernel.Λ
    even_odd = Dict(fermion => :odd, boson => :even)[statistics]
    uhat = hat.(û_base, even_odd, 0:length(u)-1; n_asymp=conv_radius) # TODO: fix this

    return FiniteTempBasis(kernel, (u, s, v), statistics, β, u_, v_, s_, uhat)
end

Base.firstindex(::AbstractBasis) = 1
Base.length(basis::AbstractBasis) = length(basis.s)

"""
    iswellconditioned(basis)

Return `true` if the sampling is expected to be well-conditioned.
"""
iswellconditioned(::IRBasis) = true
iswellconditioned(::FiniteTempBasis) = true

function Base.getindex(basis::FiniteTempBasis, i)
    u, s, v = basis.sve_result
    sve_result = u[i], s[i], v[i]
    return FiniteTempBasis(basis.statistics, beta(basis), wmax(basis); kernel=basis.kernel,
                           sve_result)
end

"""
    wmax(basis::FiniteTempBasis)

Real frequency cutoff.
"""
wmax(basis::FiniteTempBasis) = basis.kernel.Λ / beta(basis)

"""
    finite_temp_bases(β, wmax, ε, sve_result=compute(LogisticKernel(β * wmax); ε))

Construct FiniteTempBasis objects for fermion and bosons using the same LogisticKernel instance.
"""
function finite_temp_bases(β, wmax, ε, sve_result=compute(LogisticKernel(β * wmax); ε))
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
    _default_matsubara_sampling_points(basis; mitigate=true)

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
    wn = extrema(polyhat)

    # While the condition number for sparse sampling in tau saturates at a
    # modest level, the conditioning in Matsubara steadily deteriorates due
    # to the fact that we are not free to set sampling points continuously.
    # At double precision, tau sampling is better conditioned than iwn
    # by a factor of ~4 (still OK). To battle this, we fence the largest
    # frequency with two carefully chosen oversampling points, which brings
    # the two sampling problems within a factor of 2.
    if mitigate
        wn_outer = [first(wn), last(wn)]
        wn_diff = 2 * round.(Int, 0.025 * wn_outer)
        length(wn) ≥ 20 && append!(wn, wn_outer - wn_diff)
        length(wn) ≥ 42 && append!(wn, wn_outer + wn_diff)
        unique!(wn)
    end

    if iseven(first(wn))
        pushfirst!(wn, 0)
        unique!(wn)
    end

    return wn
end

function _get_kernel(statistics, Λ, kernel)
    statistics ∈ (fermion, boson) ||
        error("""statistics must be either boson (for fermionic basis) or fermion (for bosonic basis)""")
    if isnothing(kernel)
        kernel = LogisticKernel(Λ)
    else
        @assert kernel.Λ ≈ Λ
    end
    return kernel
end
