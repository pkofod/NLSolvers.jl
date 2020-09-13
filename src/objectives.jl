abstract type ObjWrapper end

"""
    ScalarObjective

"""
struct ScalarObjective{Tf, Tg, Tfg, Tfgh, Th, Thv, Tbf}
    f::Tf
    g::Tg
    fg::Tfg
    fgh::Tfgh
    h::Th
    hv::Thv
    batched_f::Tbf
end
value(so::ScalarObjective, x) = so.f(x)
# need fall back for the case where fgh is not there
upto_gradient(so::ScalarObjective, x, ∇f) = so.fg(x, ∇f)
upto_hessian(so::ScalarObjective, x, ∇f, ∇²f) = so.fgh(x, ∇f, ∇²f)
has_batched_f(so::ScalarObjective) = !(so.batched_f === nothing)
"""
    batched_value(obj, X)

Return the objective evaluated at all elements of X. If obj contains
a batched_f it will have X passed collectively, else f will be broadcasted
across the elements of X.
"""
function batched_value(obj::ScalarObjective, X)
    if has_batched_f(obj)# add 
      return obj.batched_f(X)
    else
      return obj.f.(X)
    end
  end
  function batched_value(obj::ScalarObjective, F, X)
    if has_batched_f(obj)# add 
        F = obj.batched_f(F, X)
    else
        F .= obj.f.(X)
    end
    F
end

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


## If prob is a NEqProblem, then we can just dispatch to least squares MeritObjective
# if fast JacVec exists then maybe even line searches that updates the gradient can be used??? 
struct LineObjective!{TP, T1, T2, T3}
    prob::TP
    ∇fz::T1
    z::T2
    x::T2
    d::T2
    φ0::T3
    dφ0::T3
end
(le::LineObjective!)(λ)=value(le.prob, retract!(_manifold(le.prob), le.z, le.x, le.d, λ))
function (le::LineObjective!)(λ, calc_grad::Bool)
    f, g = upto_gradient(le.prob, retract!(_manifold(le.prob), le.z, le.x, le.d, λ), le.∇fz)
    f, dot(g, le.d)
end
struct LineObjective{TP, T1, T2, T3}
    prob::TP
    ∇fz::T1
    z::T2
    x::T2
    d::T2
    φ0::T3
    dφ0::T3
end
(le::LineObjective)(λ)=value(le.prob, retract(_manifold(le.prob), le.x, le.d, λ))
function (le::LineObjective)(λ, calc_grad::Bool)
    f, g = upto_gradient(le.prob, retract(_manifold(le.prob), le.x, le.d, λ), le.∇fz)
    f, dot(g, le.d)
end

# We call real on dφ0 because x and df might be complex
_lineobjective(mstyle::InPlace, prob::AbstractProblem, ∇fz, z, x, d, φ0, dφ0) = LineObjective!(prob, ∇fz, z, x, d, φ0, real(dφ0))
_lineobjective(mstyle::OutOfPlace, prob::AbstractProblem, ∇fz, z, x, d, φ0, dφ0) = LineObjective(prob, ∇fz, z, x, d, φ0, real(dφ0))

struct MeritObjective{TP, T1, T2, T3, T4, T5}
  prob::TP
  F::T1
  FJ::T2
  Fx::T3
  Jx::T4
  d::T5
end
function value(mo::MeritObjective, x)
  Fx = mo.F(x, mo.Fx)
  (norm(Fx)^2)/2
end

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