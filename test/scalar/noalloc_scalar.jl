using NLSolvers, Test

function fourth(x, ∇f, ∇²f)
    if !(∇²f == nothing)
        ∇²f = 12x^2 - sin(x)
    end
    if !(∇f == nothing)
        ∇f = 4x^3 + cos(x)
    end

    fx = x^4 + sin(x)
    if ∇f == nothing && ∇²f == nothing
        return fx
    elseif ∇²f == nothing
        return fx, ∇f
    else
        return fx, ∇f, ∇²f
    end
end


@testset "scalar no-alloc" begin
scalar_obj = TwiceDiffed(fourth; infer=true)
_alloc = @allocated minimize(scalar_obj, 4.0, LineSearch(SR1(Direct())), MinOptions())
_alloc = @allocated minimize(scalar_obj, 4.0, LineSearch(SR1(Direct())), MinOptions())
@test _alloc == 0

_alloc = @allocated minimize(scalar_obj, 4.0, LineSearch(BFGS(Direct())), MinOptions())
_alloc = @allocated minimize(scalar_obj, 4.0, LineSearch(BFGS(Direct())), MinOptions())
@test _alloc == 0

_alloc = @allocated minimize(scalar_obj, 4.0, LineSearch(DFP(Direct())), MinOptions())
_alloc = @allocated minimize(scalar_obj, 4.0, LineSearch(DFP(Direct())), MinOptions())
@test _alloc == 0

_alloc = @allocated minimize(scalar_obj, 4.0, LineSearch(Newton()), MinOptions())
_alloc = @allocated minimize(scalar_obj, 4.0, LineSearch(Newton()), MinOptions())
@test _alloc == 0

_alloc = @allocated minimize(scalar_obj, 4.0, LineSearch(Newton()), MinOptions())
_alloc = @allocated minimize(scalar_obj, 4.0, LineSearch(Newton()), MinOptions())
@test _alloc == 0
end
