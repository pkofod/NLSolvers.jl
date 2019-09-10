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

using SparseDiffTools

import Base: show

include("objectives.jl")
export NonDiff, OnceDiff, TwiceDiff

# make this struct that has scheme and approx
abstract type QuasiNewton{T1} end

abstract type HessianApproximation end
struct Inverse <: HessianApproximation end
struct Direct <: HessianApproximation end

struct OptOptions{T1, T2}
    g_tol::T1
    maxiter::T2
    show_trace::Bool
end

OptOptions(; g_tol=1e-8, maxiter=10000, show_trace=false) =
OptOptions(g_tol, maxiter, show_trace)

abstract type LineSearch end
include("linesearches/root.jl")

# Include the actual functions that expose the functionality in this package.
include("directsearch/directsearch.jl")
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
export Newton
export ResidualKrylov, ResidualKrylovProblem
# Forcing Terms
export FixedForceTerm, DemboSteihaug, EisenstatWalkerA, EisenstatWalkerB

export NLEProblem, OptProblem
end # module
