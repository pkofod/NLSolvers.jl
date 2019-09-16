using NLSolvers, Test

function fourth(∇²f, ∇f, x)
    if !(∇²f == nothing)
        ∇²f = 12x^2 - sin(x)
    end
    if !(∇f == nothing)
        ∇f = 4x^3 + cos(x)
    end

    fx = x^4
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
_alloc = @allocated minimize(scalar_obj, 4.0, SR1(Direct()))
_alloc = @allocated minimize(scalar_obj, 4.0, SR1(Direct()))
@test _alloc == 0

_alloc = @allocated minimize(scalar_obj, 4.0, BFGS(Direct()))
_alloc = @allocated minimize(scalar_obj, 4.0, BFGS(Direct()))
@test _alloc == 0

_alloc = @allocated minimize(scalar_obj, 4.0, DFP(Direct()))
_alloc = @allocated minimize(scalar_obj, 4.0, DFP(Direct()))
@test _alloc == 0

_alloc = @allocated minimize(scalar_obj, 4.0, Newton(Direct()))
_alloc = @allocated minimize(scalar_obj, 4.0, Newton(Direct()))
@test _alloc == 0

_alloc = @allocated minimize(scalar_obj, 4.0, Newton(Direct()))
_alloc = @allocated minimize(scalar_obj, 4.0, Newton(Direct()))
@test _alloc == 0
end
