function minimize!(objective::ObjWrapper, x0, scheme, options=OptOptions())
    minimize!(objective, (x0, nothing), (scheme, NWI()), options)
end
function minimize!(objective::ObjWrapper, x0, approach::Tuple{<:Any, <:TRSPSolver}, options=OptOptions())
    minimize!(objective, (x0, nothing), approach, options)
end
function minimize!(objective::ObjWrapper, state0::Tuple, scheme, options=OptOptions())
    minimize!(objective, state0, (scheme, NWI()), options)
end
function minimize!(objective::ObjWrapper, state0::Tuple, approach::Tuple{<:Any, <:TRSPSolver}, options=OptOptions())
    tr_minimize!(objective, state0, approach, options)
end
include("optimize/inplace_loop.jl")
include("optimize/outofplace_loop.jl")
