solve!(prob::MinProblem{<:Any, <:Nothing, <:Nothing, <:Nothing}, x0, scheme, options::MinOptions) =
  minimize!(prob.objective, (x0, nothing), (scheme, NWI()), options)

solve!(prob::MinProblem{<:Any, <:Nothing, <:Nothing, <:Nothing}, x0, approach::Tuple{<:Any, <:TRSPSolver}, options=MinOptions()) =
  minimize!(objective, (x0, nothing), approach, options)

solve!(prob::MinProblem{<:Any, <:Nothing, <:Nothing, <:Nothing}, state0::Tuple, approach::Tuple{<:QuasiNewton, <:TRSPSolver}, options=MinOptions()) =
  tr_minimize!(objective, state0, approach, options)

function minimize!(objective::ObjWrapper, x0, scheme, options=MinOptions())
    minimize!(objective, (x0, nothing), (scheme, NWI()), options)
end
function minimize!(objective::ObjWrapper, x0, approach::Tuple{<:Any, <:TRSPSolver}, options=MinOptions())
    minimize!(objective, (x0, nothing), approach, options)
end
function minimize!(objective::ObjWrapper, state0::Tuple, approach::Tuple{<:QuasiNewton, <:TRSPSolver}, options=MinOptions())
    tr_minimize!(objective, state0, approach, options)
end
include("optimize/inplace_loop.jl")
include("optimize/outofplace_loop.jl")
