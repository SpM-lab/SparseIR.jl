using SparseIR

function test(n)
    u, s, v = compute(LogisticKernel(42.0))

    # Keep only even number of polynomials
    u, s, v = u[begin:(end - end % 2)], s[begin:(end - end % 2)],
              v[begin:(end - end % 2)]

    return overlap(u[1], u[1:n])
end

##

@profview test(6)

##

using Pkg
Pkg.activate(".")
using SparseIR
using ProfileView
function test(n)
    u, s, v = compute(LogisticKernel(42.0))

    # Keep only even number of polynomials
    u, s, v = u[begin:(end - end % 2)], s[begin:(end - end % 2)],
              v[begin:(end - end % 2)]

    return overlap(u[1], u[1:n])
end
@profview test(7)
