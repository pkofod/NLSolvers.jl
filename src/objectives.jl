abstract type ObjWrapper end

########
## C⁰ ##
########
struct NonDiffed{Tobj} <: ObjWrapper
    obj::Tobj
    NonDiffed(f; infer::Bool) = new{infer ? typeof(f) : Any}(f)
end
(od::NonDiffed)(x) = od.obj(x)
(od::NonDiffed)(x, arg, args...) = throw(ArgumentError("NonDiffed cannot be called with more than one arguement."))

########
## C¹ ##
########
struct OnceDiffed{Tobj, TAD} <: ObjWrapper
    obj::Tobj
    source::TAD
    function OnceDiffed(f; infer=false)
        source = nothing # no AD gradient
        Tsource = Nothing # no AD gradient
        inferred = infer ? typeof(f) : Any
        return new{inferred, Nothing}(f, source)
    end
end
derivative_source(od::OnceDiffed{<:Any, Nothing}) = "user supplied"
(od::OnceDiffed)(x) = od.obj(nothing, x)
(od::OnceDiffed)(x, ∇f) = od.obj(∇f, x)
(od::OnceDiffed)(x, ∇f, args...) = throw(ArgumentError("OnceDiffed cannot be called with more than two arguements."))

########
## C² ##
########
struct TwiceDiffed{Tobj, TAD} <: ObjWrapper
    obj::Tobj
    source::TAD
    function TwiceDiffed(f; infer=false)
        source = nothing # no AD gradient
        Tsource = Nothing
        inferred = infer ? typeof(f) : Any
        return new{inferred, Tsource}(f, source)
    end
end
(td::TwiceDiffed)(x) = td.obj(nothing, nothing, x)
(td::TwiceDiffed)(x, ∇f) = td.obj(nothing, ∇f, x)
(td::TwiceDiffed)(x, ∇f, ∇²f) = td.obj(∇²f, ∇f, x)
