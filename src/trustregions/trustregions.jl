
function minimize!(objective::ObjWrapper, x0, scheme::Newton, B0=nothing, options=OptOptions())
    minimize!(objective, x0, (scheme, NWI()), B0, options)
end
function minimize!(objective::ObjWrapper, x0, approach::Tuple{<:Any, <:TRSPSolver}, B0=nothing, options=OptOptions())
    tr_minimize!(objective, x0, approach, B0, options)
end
include("optimize/inplace_loop.jl")
include("optimize/outofplace_loop.jl")
