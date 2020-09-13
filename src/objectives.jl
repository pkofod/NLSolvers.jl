abstract type ObjWrapper end

"""
    ScalarObjective

"""
struct ScalarObjective{Tf, Tg, Tfg, Tfgh, Th, Thv, Tbf, P}
    f::Tf
    g::Tg
    fg::Tfg
    fgh::Tfgh
    h::Th
    hv::Thv
    batched_f::Tbf
    param::P
end
has_param(so::ScalarObjective) = so.param === nothing ? false : true
function value(so::ScalarObjective, x)
    if has_param(so)
        return so.f(x, so.param)
    else  
        return so.f(x)
    end
end
# need fall back for the case where fgh is not there
function upto_gradient(so::ScalarObjective, x, ∇f)
    if has_param(so)
        return so.fg(x, ∇f, so.param)
    else
        return so.fg(x, ∇f)
    end
end
function upto_hessian(so::ScalarObjective, x, ∇f, ∇²f)
    if has_param(so)
        return so.fgh(x, ∇f, ∇²f, so.param)
    else
        return so.fgh(x, ∇f, ∇²f)
    end
end
has_batched_f(so::ScalarObjective) = !(so.batched_f === nothing)
"""
    batched_value(obj, X)

Return the objective evaluated at all elements of X. If obj contains
a batched_f it will have X passed collectively, else f will be broadcasted
across the elements of X.
"""
function batched_value(obj::ScalarObjective, X)
    if has_batched_f(obj)
        if has_param(so)
            return obj.batched_f(X, so.param)
        else
            return obj.batched_f(X)
        end
    else
        if has_param(so)
            return obj.f.(X, Ref(so.param))
        else
            return obj.f.(X)
        end
    end
end
function batched_value(obj::ScalarObjective, F, X)
    if has_batched_f(obj)# add
        if has_param(so)
            return F = obj.batched_f(F, X, so.param)
        else
            return F = obj.batched_f(F, X)
        end
    else
        if has_param(so)
            F .= obj.f.(X, Ref(so.param))
            return F
        else
            F .= obj.f.(X, Ref(so.param))
            return F
        end
    end
end

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