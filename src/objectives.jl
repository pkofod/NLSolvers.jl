abstract type ObjWrapper end
########
## C⁰ ##
########
struct NonDiffed{Tobj} <: ObjWrapper
    obj::Tobj
    NonDiffed(f, infer=nothing) = new{infer !== nothing ? typeof(f) : Any}(f)
end
(od::NonDiffed)(x...) = od.obj(x...)

########
## C¹ ##
########
struct OnceDiffed{Tobj} <: ObjWrapper
    obj::Tobj
    function OnceDiffed(f, infer=nothing)
        inferred = infer !== nothing ? typeof(f) : Any
        return new{inferred}(f)
    end
end
(od::OnceDiffed)(x) = od.obj(x, nothing)
(od::OnceDiffed)(x, ∇f) = od.obj(x, ∇f)
(od::OnceDiffed)(x, F, J) = od.obj(x, F, J)
(od::OnceDiffed)(x, y, z, args...) = throw(ArgumentError("OnceDiffed cannot be called with more than three arguements."))

struct OnceDiffedJv{TR, TJv} <: ObjWrapper
    R::TR # residual
    Jv::TJv # jacobian vector product operator
end
(od::OnceDiffedJv)(x) = od.R(x, nothing)
(od::OnceDiffedJv)(x, ∇f) = od.R(x, ∇f)
(od::OnceDiffedJv)(x, F, J) = od.R(x, F, J)
(od::OnceDiffedJv)(x, y, z, args...) = throw(ArgumentError("OnceDiffedJv cannot be called with more than three arguements."))
########
## C² ##
########
struct TwiceDiffed{Tobj} <: ObjWrapper
    obj::Tobj
    function TwiceDiffed(f, infer=nothing)
        inferred = infer !== nothing ? typeof(f) : Any
        return new{inferred}(f)
    end
end
(td::TwiceDiffed)(x) = td.obj(x, nothing, nothing)
(td::TwiceDiffed)(x, ∇f) = td.obj(x, ∇f, nothing)
(td::TwiceDiffed)(x, ∇f, ∇²f) = td.obj(x, ∇f, ∇²f)


struct LineObjective!{TP, T1, T2, T3, T4, T5}
    prob::TP
    obj::T1
    ∇fz::T2
    z::T3
    x::T3
    d::T4
    φ0::T5
    dφ0::T5
end
(le::LineObjective!)(λ)=(le.obj(retract!(_manifold(le.prob), le.z, le.x, le.d, λ)))
function (le::LineObjective!)(λ, calc_grad::Bool)
    f, g = (le.obj(retract!(_manifold(le.prob), le.z, le.x, le.d, λ), le.∇fz))
    f, dot(g, le.d)
end
struct LineObjective{TP, T1, T2, T3, T4, T5}
    prob::TP
    obj::T1
    ∇fz::T2
    z::T3
    x::T3
    d::T4 # could be T3 i believe
    φ0::T5
    dφ0::T5
end
(le::LineObjective)(λ)=(le.obj(retract(_manifold(le.prob), le.x, le.d, λ)))
function (le::LineObjective)(λ, calc_grad::Bool)
    f, g = le.obj(retract(_manifold(le.prob), le.x, le.d, λ), le.∇fz)
    f, dot(g, le.d)
end

# We call real on dφ0 because x and df might be complex
_lineobjective(mstyle::InPlace, prob::AbstractProblem, obj::ObjWrapper, ∇fz, z, x, d, φ0, dφ0) = LineObjective!(prob, obj, ∇fz, z, x, d, φ0, real(dφ0))
_lineobjective(mstyle::OutOfPlace, prob::AbstractProblem, obj::ObjWrapper, ∇fz, z, x, d, φ0, dφ0) = LineObjective(prob, obj, ∇fz, z, x, d, φ0, real(dφ0))

struct MeritObjective{TP, T1, T2, T3, T4}
  prob::TP
  F::T1
  Fx::T2
  Jx::T3
  d::T4
end
function (mo::MeritObjective)(x)
  Fx = mo.F(x, mo.Fx, nothing) # FIXME FIXME should not have to include J here
  (norm(Fx)^2)/2
end
function (mo::MeritObjective)(x, calc_grad::Bool)
  Fx, Jx = mo.F(x, mo.Fx, mo.Jx) # FIXME FIXME should not have to include J here
  (norm(Fx)^2)/2, dot(Fx, Jx*mo.d)
end
# (_y,_x)->norm(F(_x, Fx, Jx)[1])^2)
struct Batched{T<:ObjWrapper}
  obj::T
end
isbatched(obj) = false
isbatched(obj::Batched) = true
function batched_value(prob, F, X)
  if isbatched(prob.objective)

    prob.objective.obj.obj(F, X)
  else
    for i in eachindex(F, X)
      @inbounds F[i] = prob.objective(X[i])
    end 
  end
  F
end
(odbo::Batched)(args...) = odbo.obj(args...)

struct LsqWrapper{Tobj, TF, TJ} <: ObjWrapper
  R::Tobj
  F::TF
  J::TJ
end
function (lw::LsqWrapper)(x)
  F = lw.R(x, lw.F)
  sum(abs2, F)/2
end
function (lw::LsqWrapper)(x, ∇f)
  _F, _J = lw.R(x, lw.F, lw.J)
  copyto!(∇f, sum(_J; dims=1))
  sum(abs2, _F), ∇f
end