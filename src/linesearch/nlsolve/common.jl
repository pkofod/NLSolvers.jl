struct NLEProblem{T}
    F::T
end
function value(nleq::NLEProblem, x)
    nleq.F(x)
end

struct Static
end

struct NLEOptions{T, Tmi}
    f_abstol::T
    f_reltol::T
    maxiter::Tmi
end
NLEOptions(; f_abstol=1e-8, f_reltol=1e-8, maxiter=10^4) = NLEOptions(f_abstol, f_reltol, maxiter)

include("secant.jl")
#include("inplace_loop.jl")
#include("outofplace_loop.jl")
