abstract type ObjWrapper end

struct NonDiff{Tobj} <: ObjWrapper
    obj::Tobj
end

NonDiff(f; infer=false) = NonDiff{infer ? typeof(f) : Any}(f)
(od::NonDiff)(x) = od.obj(x)
(od::NonDiff)(x, arg, args...) = throw(ArgumentError("NonDiff cannot be called with more than one arguement."))

struct OnceDiff{Tobj, TAD} <: ObjWrapper
    obj::Tobj
    source::TAD
end

function inferOnceDiff(f, source, infer)
    if infer
        # Just pass it on
        return OnceDiff(f, source)
    else
        return OnceDiff{Any, Nothing}(f, source)
    end
end
function OnceDiff(f; infer=false)
    source = nothing # no AD gradient
    inferOnceDiff(f, source, infer)
end
derivative_source(od::OnceDiff{<:Any, Nothing}) = "user supplied"
function OnceDiff(f::NonDiff, x; infer=false)
    δ = OnceFromNon(f, x)
    OnceDiff(δ, )
end
(od::OnceDiff)(x) = od.obj(nothing, x)
(od::OnceDiff)(x, ∇f) = od.obj(∇f, x)

struct TwiceDiff{Tobj, TAD} <: ObjWrapper
    obj::Tobj
    source::TAD
end
function TwiceDiff(f; infer=false)
    source = nothing # no AD gradient
    if infer
        # Just pass it on
        return TwiceDiff(f, source)
    else
        return TwiceDiff{Any, Nothing}(f, source)
    end
end
(td::TwiceDiff)(x) = td.obj(nothing, nothing, x)
(td::TwiceDiff)(x, ∇f) = td.obj(nothing, ∇f, x)
(td::TwiceDiff)(x, ∇f, ∇²f) = td.obj(∇²f, ∇f, x)
