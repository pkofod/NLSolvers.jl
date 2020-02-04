abstract type ObjWrapper end
_manifold(ow::ObjWrapper) = ow.manifold
########
## C⁰ ##
########
struct NonDiffed{Tobj, Tman<:Manifold} <: ObjWrapper
    obj::Tobj
    manifold::Tman
    NonDiffed(f, manifold=Euclidean(0); infer::Bool=false) = new{infer ? typeof(f) : Any, typeof(manifold)}(f, manifold)
end
(od::NonDiffed)(x) = od.obj(x)
(od::NonDiffed)(x, arg, args...) = throw(ArgumentError("NonDiffed cannot be called with more than one arguement."))

########
## C¹ ##
########
struct OnceDiffed{Tobj, Tman<:Manifold, TAD} <: ObjWrapper
    obj::Tobj
    manifold::Tman
    source::TAD
    function OnceDiffed(f, manifold=Euclidean(0); infer=false)
        source = nothing # no AD gradient
        Tsource = Nothing # no AD gradient
        inferred = infer ? typeof(f) : Any
        return new{inferred, typeof(manifold), Nothing}(f, manifold, source)
    end
end
derivative_source(od::OnceDiffed{<:Any, <:Any, Nothing}) = "user supplied"
(od::OnceDiffed)(x) = od.obj(nothing, x)
(od::OnceDiffed)(x, ∇f) = od.obj(∇f, x)
(od::OnceDiffed)(x, F, J) = od.obj(J, F, x)
(od::OnceDiffed)(x, y, z, args...) = throw(ArgumentError("OnceDiffed cannot be called with more than three arguements."))

struct OnceDiffedJv{TR, TJv, Tman<:Manifold, TAD} <: ObjWrapper
    R::TR # residual
    Jv::TJv # jacobian vector product operator
    manifold::Tman
    source::TAD
    function OnceDiffedJv(f, jv, manifold=Euclidean(0); infer=false)
        source = nothing # no AD gradient
        Tsource = Nothing # no AD gradient
        inferred_r = infer ? typeof(f) : Any
        inferred_jv = infer ? typeof(jv) : Any
        return new{inferred_r, inferred_jv, typeof(manifold), Nothing}(f, jv, manifold, source)
    end
end
derivative_source(od::OnceDiffedJv{<:Any, <:Any, <:Any, Nothing}) = "user supplied"
(od::OnceDiffedJv)(x) = od.R(nothing, x)
(od::OnceDiffedJv)(x, ∇f) = od.R(∇f, x)
(od::OnceDiffedJv)(x, F, J) = od.R(J, F, x)
(od::OnceDiffedJv)(x, y, z, args...) = throw(ArgumentError("OnceDiffedJv cannot be called with more than three arguements."))

########
## C² ##
########
struct TwiceDiffed{Tobj, Tman<:Manifold, TAD} <: ObjWrapper
    obj::Tobj
    manifold::Tman
    source::TAD
    function TwiceDiffed(f, manifold=Euclidean(0); infer=false)
        source = nothing # no AD gradient
        Tsource = Nothing
        inferred = infer ? typeof(f) : Any
        return new{inferred, typeof(manifold), Tsource}(f, manifold, source)
    end
end
(td::TwiceDiffed)(x) = td.obj(nothing, nothing, x)
(td::TwiceDiffed)(x, ∇f) = td.obj(nothing, ∇f, x)
(td::TwiceDiffed)(x, ∇f, ∇²f) = td.obj(∇²f, ∇f, x)


struct LineObjective!{T1, T2, T3, T4, T5}
    obj::T1
    ∇fz::T2
    z::T3
    x::T3
    d::T4
    φ0::T5
    dφ0::T5
end
(le::LineObjective!)(λ)=(le.obj(retract!(_manifold(le.obj), le.z, le.x, le.d, λ)))
function (le::LineObjective!)(λ, calc_grad::Bool)
    f, g = (le.obj(retract!(_manifold(le.obj), le.z, le.x, le.d, λ), le.∇fz))
    f, dot(g, le.d)
end
struct LineObjective{T1, T2, T3, T4, T5}
    obj::T1
    ∇fz::T2
    z::T3
    x::T3
    d::T4
    φ0::T5
    dφ0::T5
end
(le::LineObjective)(λ)=(le.obj(retract(_manifold(le.obj), le.x, le.d, λ)))
function (le::LineObjective)(λ, calc_grad::Bool)
    f, g = le.obj(retract(_manifold(le.obj), le.x, le.d, λ), le.∇fz)
    f, dot(g, le.d)
end

_lineobjective(mstyle::InPlace, obj, ∇fz, z, x, d, φ0, dφ0) = LineObjective!(obj, ∇fz, z, x, d, φ0, dφ0)
_lineobjective(mstyle::OutOfPlace, obj, ∇fz, z, x, d, φ0, dφ0) = LineObjective(obj, ∇fz, z, x, d, φ0, dφ0)

struct MeritObjective{T1, T2, T3, T4}
  F::T1
  Fx::T2
  Jx::T3
  d::T4
end
_manifold(mo::MeritObjective) = mo.F.manifold
function (mo::MeritObjective)(x)
  Fx = mo.F(x, mo.Fx, nothing) # FIXME FIXME should not have to include J here
  (norm(Fx)^2)/2
end
function (mo::MeritObjective)(x, calc_grad::Bool)
  Fx, Jx = mo.F(x, mo.Fx, mo.Jx) # FIXME FIXME should not have to include J here
  (norm(Fx)^2)/2, dot(Fx, Jx*mo.d)
end
# (_y,_x)->norm(F(_x, Fx, Jx)[1])^2)
