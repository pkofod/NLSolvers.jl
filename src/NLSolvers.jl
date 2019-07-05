module NLSolvers
# We use often use the LinearAlgebra functions dot and norm for operations rela-
# ted to assessing angles between vectors, size of vectors and so on.
using LinearAlgebra: dot, I, norm,
                     mul!, rmul!,
                     cholesky,
                     UniformScaling, Symmetric,
                     ldiv!, factorize,
                     eigen, Hermitian,
                     diag, # for trust region diagonal manipulation
                     opnorm # for NWI safe guards

using RandomNumbers # for better random number generators and also rand!


# make this struct that has scheme and approx
abstract type QuasiNewton{T1} end

struct NLLS{O}
    obj::O
end
function (nlls::NLLS)(∇f, x)
    F, G = nlls.obj(∇f, x)
    f = sum(F)
    if isa(G, Nothing)
        return f
    else
        f, G'*F
    end
    return
end

struct OptOptions{T1, T2}
    c::T1
    g_tol::T1
    maxiter::T2
    show_trace::Bool
end

OptOptions(; c=1e-4, g_tol=1e-8, maxiter=500, show_trace=false) =
OptOptions(c, g_tol, maxiter, show_trace)

abstract type HessianApproximation end
struct Inverse <: HessianApproximation end
struct Direct <: HessianApproximation end

# Include the actual functions that expose the functionality in this package.
include("linesearch/linesearch.jl")
include("quasinewton/quasinewton.jl")
include("trustregions/trustregions.jl")
include("randomsearch/randomsearch.jl")

export minimize, minimize!, OptOptions
export nlsolve, nlsolve!
export backtracking, BackTracking
export NWI, TRSolver
export Inverse, Direct
# Export algos
export BFGS, SR1, DFP, GradientDescent
export NLEProblem
end # module
