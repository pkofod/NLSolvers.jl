module NLSolvers

import Base: show

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


include("objectives.jl")
export NonDiffed, OnceDiffed, TwiceDiffed

# make this struct that has scheme and approx
abstract type QuasiNewton{T1} end

abstract type HessianApproximation end
struct Inverse <: HessianApproximation end
struct Direct <: HessianApproximation end
export Inverse, Direct

struct OptOptions{T1, T2}
    g_tol::T1
    maxiter::T2
    show_trace::Bool
end
export OptOptions

OptOptions(; g_tol=1e-8, maxiter=10000, show_trace=false) =
OptOptions(g_tol, maxiter, show_trace)

# Globalization strategies
abstract type LineSearch end
include("globalization/linesearches/root.jl")
export backtracking, Backtracking, TwoPointQuadratic
include("globalization/trs_solvers/root.jl")
export NWI#, TRSolver

# Quasi-Newton (including Newton and gradient descent) functionality
include("quasinewton/quasinewton.jl")
export BFGS, SR1, DFP, GradientDescent, Newton

# Include the actual functions that expose the functionality in this package.
include("optimize/linesearch/linesearch.jl")
include("optimize/randomsearch/randomsearch.jl")
include("optimize/directsearch/directsearch.jl")
include("optimize/trustregions/trustregions.jl")
export minimize, minimize!, OptProblem

include("nlsolve/root.jl")
export nlsolve, nlsolve!, NLEProblem
export ResidualKrylov, ResidualKrylovProblem
# Forcing Terms
export FixedForceTerm, DemboSteihaug, EisenstatWalkerA, EisenstatWalkerB
end # module
