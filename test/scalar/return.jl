using NLSolvers, Test, DoubleFloats
function myfun(∇f, x)
    myfun(nothing, ∇f, x)
end
function myfun(∇²f, ∇f, x::T) where T
    if !(∇²f == nothing)
        ∇²f = 12x^2 - sin(x)
    end
    if !(∇f == nothing)
        ∇f = 4x^3 + cos(x)
    end

    fx = x^4 +sin(x)
    if ∇f == nothing && ∇²f == nothing
        return T(fx)
    elseif ∇²f == nothing
        return T(fx), T(∇f)
    else
        return T(fx), T(∇f), T(∇²f)
    end
end
@testset "scalar return types" begin
for T = (Float16, Float32, Float64, Rational{BigInt}, Double32, Double64)
    for M in (SR1, BFGS, DFP, Newton)
        if M == Newton
            obj = TwiceDiff(myfun)
        else
            obj = OnceDiff(myfun)
        end
        res = minimize(obj, T(4), M(Direct()))
        @test all(isa.(res[1:3], T))
        res = minimize(obj, T(4), M(Inverse()))
        @test all(isa.(res[1:3], T))
    end
end
end
