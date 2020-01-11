# [SOREN] TRUST REGION MODIFICATION OF NEWTON'S METHOD
# [N&W] Numerical optimization
# [Yuan] A review of trust region algorithms for optimization
abstract type TRSPSolver end
abstract type NearlyExactTRSP <: TRSPSolver end

include("solvers/NWI.jl")
include("solvers/dogleg.jl")
include("solvers/NTR.jl")
#include("subproblemsolvers/TRS.jl") just make an example instead of relying onTRS.jl
