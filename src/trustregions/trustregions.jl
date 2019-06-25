# [SOREN] TRUST REGION MODIFICATION OF NEWTON'S METHOD
# [N&W] Numerical optimization
# [Yuan] A review of trust region algorithms for optimization
abstract type TRSPSolver end
abstract type NearlyExactTRSP <: TRSPSolver end

include("subproblemsolvers/NWI.jl")
#include("subproblemsolvers/TRS.jl") just make an example instead of relying onTRS.jl
function minimize!(objective, x0, scheme::Newton, B0=nothing, options=OptOptions())
    minimize!(objective, x0, (scheme, NWI()), B0, options)
end
function minimize!(objective, x0, approach::Tuple{<:Any, <:TRSPSolver}, B0=nothing, options=OptOptions())
    tr_minimize!(objective, x0, approach, B0, options)
end
include("optimize/inplace_loop.jl")
include("optimize/outofplace_loop.jl")
