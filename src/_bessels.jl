import GSL: hypergeom, sf_gamma

besselj(α, x) = (x / 2)^α / sf_gamma(α + 1) * hypergeom([], α + 1, -x^2 / 4)
function sphericalbesselj(n, x::AbstractFloat)
    if iszero(x)
        return iszero(n) ? one(x) : zero(x)
    else
        return √(π / 2x) * besselj(n + 0.5, x)
    end
end