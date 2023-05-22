# FIXME: These are piracy, but needed to make MultiFloats work for us.
Base.Math.hypot(x::Float64x2, y::Float64x2) = Base.Math._hypot(x, y)

function Base.sinh(x::Float64x2)
    if iszero(x) || isnan(x) || isinf(x)
        return x
    end

    return if abs(x) > log(2)
        exp_x = exp(x)
        exp_minus_x = 1 / exp_x
        (exp_x - exp_minus_x) / 2
    else
        term = x
        sum = x
        n = 1
        while true
            term *= x^2 / (2n * (2n + 1))
            sum_new = sum + term
            sum == sum_new && break
            sum = sum_new
            n += 1
        end
        sum
    end
end

function Base.cosh(x::Float64x2)
    iszero(x) && return one(x)
    isnan(x) && return x
    isinf(x) && return abs(x)

    return if abs(x) > log(2)
        exp_x = exp(x)
        exp_minus_x = 1 / exp_x
        (exp_x + exp_minus_x) / 2
    else
        term = one(x)
        sum = one(x)
        n = 1
        while true
            term *= x^2 / ((2n - 1) * 2n)
            sum_new = sum + term
            sum == sum_new && break
            sum = sum_new
            n += 1
        end
        sum
    end
end
