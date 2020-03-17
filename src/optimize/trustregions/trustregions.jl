abstract type TrustRegionUpdater end
struct TrustRegion{M, SP, D}
	scheme::M
	spsolve::SP
	Δupdate::D
end
summary(tr::TrustRegion) = "$(summary(modelscheme(tr))) with $(summary(algorithm(tr)))"
function initial_preconditioner(approach::TrustRegion, x)
  nothing
end
"""
  BTR() <: TrustRegionUpdater

Basic trust region updater following, and named after [CGT].
"""
struct BTR{T}
  Δmin::T
end
TrustRegion(; deltamin=nothing) = TrustRegion(Newton(), NTR(), BTR(deltamin))
TrustRegion(m, sp=NTR(); deltamin=nothing) = TrustRegion(m, sp, BTR(deltamin))
modelscheme(tr::TrustRegion) = tr.scheme
algorithm(tr::TrustRegion) = tr.spsolve

solve!(prob::MinProblem{<:Any, <:Nothing, <:Nothing, <:Nothing}, x0, scheme, options::MinOptions) =
  minimize!(prob.objective, (x0, nothing), TrustRegion(scheme, NWI()), options)

solve!(prob::MinProblem{<:Any, <:Nothing, <:Nothing, <:Nothing}, x0, approach::TrustRegion, options::MinOptions) =
  minimize!(objective, (x0, nothing), approach, options)

solve!(prob::MinProblem{<:Any, <:Nothing, <:Nothing, <:Nothing}, s0::Tuple, approach::TrustRegion, options::MinOptions) =
  minimize!(objective, s0, approach, options)

function minimize!(objective::ObjWrapper, x0, scheme::Newton, options::MinOptions)
    minimize!(objective, (x0, nothing), TrustRegion(scheme, NTR()), options)
end
function minimize!(objective::ObjWrapper, x0, approach::TrustRegion, options::MinOptions)
    minimize!(objective, (x0, nothing), approach, options)
end
include("optimize/inplace_loop.jl")
include("optimize/outofplace_loop.jl")
