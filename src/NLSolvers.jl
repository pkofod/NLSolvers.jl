module NLSolvers

import Base: show

#============================ LinearAlgebra ===========================
  We use often use the LinearAlgebra functions dot and norm for opera-
  tions related to assessing angles between vectors, size of vectors
  and so on.

  mul!, rmul!, ldiv!, ect can to make Array{T, N} operations as fast
  a possible.

  cholesky is impossible to live without to replace inverses

  UniformScaling (I), Symmetric, Diagonal are all usefull to handle
  shifted systems, hessians, etc

  eigen and diag are useful in trust region subprblems

  opnorm is used in the trust region subproblem safe guards to bound
  the estimate on lambda
 ============================ LinearAlgebra ===========================#

using LinearAlgebra: dot, I, norm,
                     mul!, rmul!, ldiv!,
                     cholesky, factorize,
                     UniformScaling, Symmetric, Hermitian, Diagonal,
                     diag, # for trust region diagonal manipulation
                     eigen,
                     opnorm # for NWI safe guards

# For better random number generators and rand!
using RandomNumbers


"""

"""
function objective_return(f, g, H=nothing)
  if g isa Nothing && H isa Nothing
    return f
elseif !(g isa Nothing) && H isa Nothing
    return f, g
  elseif !(g isa Nothing) && !(H isa Nothing)
    return f, g, H
  end
end
export objective_return

using StaticArrays

abstract type MutateStyle end
struct InPlace <: MutateStyle end
struct OutOfPlace <:MutateStyle end

include("Manifolds.jl")
include("objectives.jl")
include("linearalgebra.jl")
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
export NWI, Dogleg#, TRSolver

# Quasi-Newton (including Newton and gradient descent) functionality
include("quasinewton/quasinewton.jl")
export DBFGS, BFGS, SR1, DFP, GradientDescent, Newton

# Include the actual functions that expose the functionality in this package.
include("optimize/linesearch/linesearch.jl")
include("optimize/randomsearch/randomsearch.jl")
include("optimize/directsearch/directsearch.jl")
export NelderMead
include("optimize/trustregions/trustregions.jl")
export minimize, minimize!, OptProblem

include("nlsolve/root.jl")
export nlsolve, nlsolve!, NLEProblem
export ResidualKrylov, ResidualKrylovProblem
# Forcing Terms
export FixedForceTerm, DemboSteihaug, EisenstatWalkerA, EisenstatWalkerB
end # module
