abstract type ObjWrapper end

struct NonDiff{Tobj} <: ObjWrapper
    obj::Tobj
    NonDiff(f; infer::Bool) = new{infer ? typeof(f) : Any}(f)
end

(od::NonDiff)(x) = od.obj(x)
(od::NonDiff)(x, arg, args...) = throw(ArgumentError("NonDiff cannot be called with more than one arguement."))

struct OnceDiff{Tobj, TAD} <: ObjWrapper
    obj::Tobj
    source::TAD
    function OnceDiff(f; infer=false)
        source = nothing # no AD gradient
        Tsource = Nothing # no AD gradient
        inferred = infer ? typeof(f) : Any
        return new{inferred, Nothing}(f, source)
    end
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
    function TwiceDiff(f; infer=false)
        source = nothing # no AD gradient
        Tsource = Nothing
        inferred = infer ? typeof(f) : Any
        return new{inferred, Tsource}(f, source)
    end
end
(td::TwiceDiff)(x) = td.obj(nothing, nothing, x)
(td::TwiceDiff)(x, ∇f) = td.obj(nothing, ∇f, x)
(td::TwiceDiff)(x, ∇f, ∇²f) = td.obj(∇²f, ∇f, x)
